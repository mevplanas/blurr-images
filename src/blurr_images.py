import os
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import yaml
from exif import Image as ExifImage
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timedelta, timezone
from pyspark.sql import SparkSession, Row
from PIL import Image as PILImage

spark = SparkSession.builder.appName("LocalSparkApp").master("local[*]").getOrCreate()

def is_valid_image(path):
    try:
        if not os.path.exists(path):
            print(f"FAILAS NERASTAS: {path}")
            return False
        size = os.path.getsize(path)
        if size < 1024:
            print(f"FAILAS PER MAŽAS ({size} bytes): {path}")
            return False
        # Tikrinam, ar pavyksta atidaryti su PIL (tik formatui validuoti)
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Vaizdas neįskaitomas: {path}, klaida: {e}")
        return False

def get_relative_altidute(img_path: str):
    """
    Grąžina aukštį iš EXIF gps_altitude, jei yra. Jei nėra – grąžina None.
    """
    try:
        with open(img_path, 'rb') as image_file:
            img = ExifImage(image_file)
        if img.has_exif:
            if hasattr(img, "gps_altitude") and img.gps_altitude is not None:
                print(f"Aukštis rastas (EXIF GPSAltitude): {img.gps_altitude}")
                return float(img.gps_altitude)
            else:
                print("Aukštis nerastas EXIF'e")
                return None
        else:
            print("EXIF nėra")
            return None
    except Exception as e:
        print(f"Error extracting altitude from EXIF: {e}")
        return None  # Jei aukščio nėra, grąžinam None

with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

log_table_name = config["LOG_TABLE_NAME"]
idx2class = config["CLASS_NAMES"]
class2idx = {v: k for k, v in idx2class.items()}
yolo_nas = YOLO(config["YOLO_MODEL"])
classes_to_blur = config["CLASSES_TO_BLURR"]
blurr_intensity = int(config["BLURR_INTENSITY"])
input_connection_string = config["INPUT_CONNECTION_STRING"]
input_container_name = config["INPUT_CONTAINER_NAME"]

def is_blob_image(blob_name: str) -> bool:
    return blob_name.lower().endswith(('.jpg', '.jpeg', '.png'))

def pipeline() -> None:
    current_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        processed_df = spark.table(log_table_name).select("file_name")
        processed_files = set(row.file_name for row in processed_df.collect())
    except Exception as e:
        print(f"Could not read feature store table: {e}")
        processed_files = set()

    records = []
    bad_images = []

    blob_service_client = BlobServiceClient.from_connection_string(input_connection_string)
    container_client = blob_service_client.get_container_client(input_container_name)

    # Filtravimas: tik paskutines 24h nuotraukos
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    blobs = container_client.list_blobs()
    blobs = [
        blob for blob in blobs
        if is_blob_image(blob.name)
        and "00_UNSORTED" not in blob.name
        and blob.name not in processed_files
        and getattr(blob, "last_modified", None) is not None
        and blob.last_modified >= cutoff
    ]

    input_dir = os.path.join(current_dir, "..", "input")
    os.makedirs(input_dir, exist_ok=True)

    for blob in tqdm(blobs):
        base_name = os.path.basename(blob.name)
        blob_dir = blob.name.replace(base_name, "")
        local_dir = os.path.join(current_dir, "..", "input", blob_dir)
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, base_name)

        # Atsisiųsk failą
        blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=blob.name)
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Validuok failą
        if not is_valid_image(path):
            print(f"Praleidžiam blogą failą: {blob.name}")
            bad_images.append(blob.name)
            try:
                os.remove(path)
            except Exception:
                pass
            continue

        altitude = get_relative_altidute(path)
        if altitude is None:
            print(f"Praleidžiam {blob.name} – nepavyko nuskaityti aukščio")
            try:
                os.remove(path)
            except Exception:
                pass
            continue  # Aukščio nėra, skipinam nuotrauką

        height_from_ground = altitude
        blurred_count = 0

        if height_from_ground < config["MINIMUM_HEIGHT_FROM_GROUND"]:
            print(f"Blurring {blob.name}")
            hat = yolo_nas.predict(
                path,
                conf=config.get("MODEL_CONFIDENCE", 0.15),
                iou=config.get("MODEL_IOU", 0.9),
            )
            boxes = hat[0].boxes
            labels = hat[0].boxes.cls
            labels = labels.cpu().numpy()
            label_names = [idx2class[label] for label in labels]
            bboxes = boxes.xyxy

            img_cv = cv2.imread(path)
            for i, box in enumerate(bboxes):
                if label_names[i] in classes_to_blur:
                    x0, y0, x1, y1 = map(int, box)
                    img_cv[y0:y1, x0:x1] = cv2.blur(
                        img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity)
                    )
                    blurred_count += 1

            cv2.imwrite(path, img_cv)
            with open(path, "rb") as data:
                container_client.upload_blob(name=blob.name, data=data, overwrite=True)

        try:
            os.remove(path)
        except Exception:
            pass

        records.append(
            (
                blob.name,
                datetime.now(),
                True if blurred_count > 0 else False,
                int(blurred_count),
            )
        )

    print(f"Processed {len(blobs)} images.")
    print(f"Blurred {sum(r[2] for r in records)} images.")
    print(f"Blurred {sum(r[3] for r in records)} objects.")

    # Probleminių failų logas
    if bad_images:
        with open(os.path.join(current_dir, "bad_images.txt"), "w") as badf:
            badf.write("\n".join(bad_images))
        print(f"Rasta {len(bad_images)} blogų failų – žiūrėk bad_images.txt")

    # Įrašymas į log lentelę
    if records:
        rows = [
            Row(file_name=r[0], datetime=r[1], blurred=r[2], blurred_objects=r[3])
            for r in records
        ]
        records_df = spark.createDataFrame(rows)
        records_df = records_df.withColumn("file_name", records_df.file_name.cast("string"))
        records_df = records_df.withColumn("datetime", records_df.datetime.cast("timestamp"))
        records_df = records_df.withColumn("blurred", records_df.blurred.cast("boolean"))
        records_df = records_df.withColumn("blurred_objects", records_df.blurred_objects.cast("int"))
        records_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(log_table_name)

if __name__ == "__main__":
    pipeline()

import os
import cv2
import yaml
import json
from tqdm import tqdm
from exif import Image
from datetime import datetime
from PIL import Image as PILImage
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from pyspark.sql import SparkSession, Row
import subprocess
import sys

try:
    import piexif
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "piexif"])
    import piexif


spark = SparkSession.builder.appName("LocalSparkApp").master("local[*]").getOrCreate()


# Extract relative altitude from XMP metadata
def get_relative_altidute(img_path: str) -> float:
    altitude = 0.0
    try:
        with PILImage.open(img_path) as img:
            xmp = img.getxmp()
            if xmp and "xmpmeta" in xmp:
                rdf = xmp["xmpmeta"].get("RDF", {})
                desc = rdf.get("Description", {})
                if "RelativeAltitude" in desc:
                    return float(desc["RelativeAltitude"])
    except Exception as e:
        print(f"Error extracting XMP metadata from image: {e}")

    # Try EXIF GPSAltitude fallback
    try:
        exif_data = piexif.load(PILImage.open(img_path).info.get("exif", b""))
        gps = exif_data.get("GPS", {})
        altitude_raw = gps.get(piexif.GPSIFD.GPSAltitude)
        if isinstance(altitude_raw, tuple) and altitude_raw[1] != 0:
            altitude = altitude_raw[0] / altitude_raw[1]
    except Exception as e:
        print(f"Error extracting EXIF GPSAltitude: {e}")

    return altitude


def extract_exif_metadata(image_path):
    metadata = {}
    try:
        img = PILImage.open(image_path)
        exif_bytes = img.info.get("exif", None)
        if not exif_bytes:
            return "{}"
        exif_data = piexif.load(exif_bytes)
        for ifd_name, ifd_dict in exif_data.items():
            if isinstance(ifd_dict, dict):
                for tag_id, value in ifd_dict.items():
                    tag_name = piexif.TAGS[ifd_name].get(tag_id, {}).get("name", f"{tag_id}")
                    try:
                        if isinstance(value, bytes):
                            value = value.decode(errors="ignore")
                        metadata[f"{ifd_name}:{tag_name}"] = value
                    except Exception:
                        metadata[f"{ifd_name}:{tag_name}"] = str(value)
    except Exception as e:
        print(f"Failed to extract EXIF metadata: {e}")
    return json.dumps(metadata)


# Load configuration
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
    return blob_name.lower().endswith((".jpg", ".jpeg", ".png"))


def pipeline() -> None:
    current_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        processed_df = spark.table(log_table_name).select("file_name")
        processed_files = set(row.file_name for row in processed_df.collect())
    except Exception as e:
        print(f"Could not read feature store table: {e}")
        processed_files = set()

    records = []

    blob_service_client = BlobServiceClient.from_connection_string(input_connection_string)
    container_client = blob_service_client.get_container_client(input_container_name)

    blobs = container_client.list_blobs()
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]
    # blobs = [blob for blob in blobs if blob.name.startswith("test/")]
    blobs = [blob for blob in blobs if "00_UNSORTED" not in blob.name]
    blobs = [blob for blob in blobs if blob.name not in processed_files]

    input_dir = os.path.join(current_dir, "..", "input")
    os.makedirs(input_dir, exist_ok=True)

    for blob in tqdm(blobs):
        base_name = os.path.basename(blob.name)
        blob_dir = blob.name.replace(base_name, "")
        os.makedirs(os.path.join(input_dir, blob_dir), exist_ok=True)
        path = os.path.join(input_dir, blob_dir, base_name)

        # Download image
        blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=blob.name)
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Extract metadata before blur
        metadata_before = extract_exif_metadata(path)
        altitude = get_relative_altidute(path)
        altitude = get_relative_altidute(path)
        height_from_ground = altitude
        print(f"📏 Altitude for {blob.name}: {height_from_ground} m")

        height_from_ground = altitude
        blurred_count = 0
        metadata_after = None

        if height_from_ground < config["MINIMUM_HEIGHT_FROM_GROUND"]:
            print(f"Blurring {blob.name}")
            hat = yolo_nas.predict(
                path,
                conf=config.get("MODEL_CONFIDENCE", 0.15),
                iou=config.get("MODEL_IOU", 0.9),
            )
            boxes = hat[0].boxes
            labels = hat[0].boxes.cls.cpu().numpy()
            label_names = [idx2class[label] for label in labels]
            bboxes = boxes.xyxy
            img_cv = cv2.imread(path)

            for i, box in enumerate(bboxes):
                if label_names[i] in classes_to_blur:
                    x0, y0, x1, y1 = map(int, box[:4])
                    img_cv[y0:y1, x0:x1] = cv2.blur(img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity))
                    blurred_count += 1

            blurred_path = os.path.join(input_dir, blob_dir, f"{base_name}")
            blurred_blob_name = os.path.join(blob_dir, f"{base_name}").replace("\\", "/")

            # Reload EXIF from original and write to blurred
            try:
                original = PILImage.open(path)
                exif_bytes = original.info.get("exif")
                if exif_bytes:
                    # Convert blurred OpenCV image to PIL
                    blurred_pil = PILImage.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    blurred_pil.save(blurred_path, "jpeg", exif=exif_bytes)
            except Exception as e:
                print(f"⚠️ Failed to copy EXIF to blurred image: {e}")

            with open(blurred_path, "rb") as data:
                container_client.upload_blob(name=blurred_blob_name, data=data, overwrite=True)

            metadata_after = extract_exif_metadata(blurred_path)

        # Log result (blurred or not)
        records.append(
            (
                blob.name,
                datetime.now(),
                blurred_count > 0,
                blurred_count,
                metadata_before,
                metadata_after if metadata_after else json.dumps({}),
            )
        )

    print(f"Processed {len(blobs)} images.")
    print(f"Blurred {sum(r[2] for r in records)} images.")
    print(f"Blurred {sum(r[3] for r in records)} objects.")

    # Write results to log table if records exist
    if not records:
        print("⚠️ No images processed. Skipping log table write.")
        return

    rows = [
        Row(
            file_name=r[0],
            datetime=r[1],
            blurred=r[2],
            blurred_objects=r[3],
            metadata_before=r[4],
            metadata_after=r[5],
        )
        for r in records
    ]

    records_df = spark.createDataFrame(rows)

    records_df = records_df.withColumn("file_name", records_df.file_name.cast("string"))
    records_df = records_df.withColumn("datetime", records_df.datetime.cast("timestamp"))
    records_df = records_df.withColumn("blurred", records_df.blurred.cast("boolean"))
    records_df = records_df.withColumn("blurred_objects", records_df.blurred_objects.cast("int"))
    records_df = records_df.withColumn("metadata_before", records_df.metadata_before.cast("string"))
    records_df = records_df.withColumn("metadata_after", records_df.metadata_after.cast("string"))

    records_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(log_table_name)


if __name__ == "__main__":
    pipeline()

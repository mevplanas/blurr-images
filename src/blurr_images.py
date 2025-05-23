import os

# Importing the computer vision model
from ultralytics import YOLO

# IMG reading
import cv2

# Iteration tracking
from tqdm import tqdm

# YAML reading
import yaml

# Image data parsing (no longer used for altitude)
from exif import Image

# Blob storage
from azure.storage.blob import BlobServiceClient

# Datetime
from datetime import datetime

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Import Pillow for the new relative altitude function
from PIL import Image as PILImage

spark = SparkSession.builder.appName("LocalSparkApp").master("local[*]").getOrCreate()


# New function to extract relative altitude from image metadata using XMP.
def get_relative_altidute(img_path: str) -> float:
    """
    Description
    -----------
    The function extracts relative height from image metadata.
 
    Parameters
    ----------
    :param img_path : Path to the image
 
    Returns
    ----------
    :return: Relative height of the image
    """
    # Set default altitude to 0
    altitude = 0.0
    try:
        with PILImage.open(img_path) as img:
            # Extract the XMP metadata from the image
            xmp = img.getxmp()
            # Get relative height from the XMP metadata
            altitude = xmp["xmpmeta"]["RDF"]["Description"]["RelativeAltitude"]
            # Convert the altitude to a float
            altitude = float(altitude)
    except Exception as e:
        print(f"Error extracting XMP metadata from image: {e}")
    return altitude


# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Extracting the log table name
log_table_name = config["LOG_TABLE_NAME"]

# Saving the class names
idx2class = config["CLASS_NAMES"]
class2idx = {v: k for k, v in idx2class.items()}

# Getting the sg model
yolo_nas = YOLO(config["YOLO_MODEL"])

# Defining the classes to blurr
classes_to_blur = config["CLASSES_TO_BLURR"]

# Extracting the blurr intensity
blurr_intensity = int(config["BLURR_INTENSITY"])

# Extracting the input connection string and container name
input_connection_string = config["INPUT_CONNECTION_STRING"]
input_container_name = config["INPUT_CONTAINER_NAME"]


def is_blob_image(blob_name: str) -> bool:
    """
    Checks if the blob is an image
    """
    return (
        blob_name.endswith(".jpg")
        or blob_name.endswith(".png")
        or blob_name.endswith(".jpeg")
        or blob_name.endswith(".JPG")
        or blob_name.endswith(".PNG")
        or blob_name.endswith(".JPEG")
    )


# Defining the pipeline
def pipeline() -> None:
    # Infer the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Get the list of already processed file names from the feature store table.
    try:
        processed_df = spark.table(log_table_name).select("file_name")
        processed_files = set(row.file_name for row in processed_df.collect())
    except Exception as e:
        # If the table doesn't exist yet or there's an error, assume no files have been processed.
        print(f"Could not read feature store table: {e}")
        processed_files = set()

    # Initialize list to store new records for the feature store table.
    # Each record is a tuple: (file_name, timestamp, blurred, blurred_objects)
    records = []

    # Creating the connection
    blob_service_client = BlobServiceClient.from_connection_string(
        input_connection_string
    )

    # Getting the container
    container_client = blob_service_client.get_container_client(input_container_name)

    # Listing all the blobs
    blobs = container_client.list_blobs()

    # Only leaving the images
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]

    # Dropping the blobs that are in the 00_UNSORTED directory
    blobs = [blob for blob in blobs if "00_UNSORTED" not in blob.name]

    # Creating the unique blob names and only processing the ones that are not in the log
    blobs = [blob for blob in blobs if blob.name not in processed_files]

    # Creating the input directory
    input_dir = os.path.join(current_dir, "..", "input")
    os.makedirs(input_dir, exist_ok=True)

    # The logic is this:
    # For each image:
    # 1. Download the image
    # 2. Predict the bounding boxes
    # 3. If the bounding box is in the classes to blurr, blurr the image
    # 4. Save the image back to the original directory in azure blob

    for blob in tqdm(blobs):
        # Getting the base name
        base_name = os.path.basename(blob.name)

        # Getting everything except the base name from blob name
        blob_dir = blob.name.replace(base_name, "")

        # Creating the directory
        os.makedirs(os.path.join(current_dir, "..", "input", blob_dir), exist_ok=True)

        # Defining the path to the image
        path = os.path.join(current_dir, "..", "input", blob_dir, base_name)

        # Downloading the blob
        blob_client = blob_service_client.get_blob_client(
            container=input_container_name, blob=blob.name
        )
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        # Using the new function to extract the relative altitude from the image metadata.
        altitude = get_relative_altidute(path)

        # Since we now use the relative altitude, the height from ground is set directly
        height_from_ground = altitude

        # Initialize the blurred objects counter
        blurred_count = 0

        # Only blurring, if the image passes the threshold
        if height_from_ground < config["MINIMUM_HEIGHT_FROM_GROUND"]:
            print(f"Blurring {blob.name}")

            # Predicting on the downloaded image
            hat = yolo_nas.predict(
                path,
                conf=config.get("MODEL_CONFIDENCE", 0.15),
                iou=config.get("MODEL_IOU", 0.9),
            )

            # Getting the bounding boxes and labels
            boxes = hat[0].boxes
            labels = hat[0].boxes.cls
            labels = labels.cpu().numpy()
            label_names = [idx2class[label] for label in labels]

            # Extracting the xyxy coordinates
            bboxes = boxes.xyxy

            # Blurring
            img_cv = cv2.imread(path)

            for i, box in enumerate(bboxes):
                if label_names[i] in classes_to_blur:
                    # Getting the coordinates
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

                    # Converting to int
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                    # Blurring the image
                    img_cv[y0:y1, x0:x1] = cv2.blur(
                        img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity)
                    )

                    # Incrementing the blurred objects counter
                    blurred_count += 1

            # Writing the blurred image
            cv2.imwrite(path, img_cv)

            # Uploading to the same place in the blob
            with open(path, "rb") as data:
                container_client.upload_blob(name=blob.name, data=data, overwrite=True)

        # Deleting the image from local storage
        os.remove(path)

        # Record the processing details with the current timestamp.
        # 'blurred' is True if at least one object was blurred, otherwise False.
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

    # After processing all blobs, create a Spark DataFrame from the records and append to the feature store table.
    if records:
        rows = [
            Row(file_name=r[0], datetime=r[1], blurred=r[2], blurred_objects=r[3])
            for r in records
        ]
        records_df = spark.createDataFrame(rows)

        # Ensuring the datatypes:
        records_df = records_df.withColumn("file_name", records_df.file_name.cast("string"))
        records_df = records_df.withColumn("datetime", records_df.datetime.cast("timestamp"))
        records_df = records_df.withColumn("blurred", records_df.blurred.cast("boolean"))
        records_df = records_df.withColumn("blurred_objects", records_df.blurred_objects.cast("int"))

        # Appending the records to the log table
        records_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(log_table_name)


if __name__ == "__main__":
    pipeline()

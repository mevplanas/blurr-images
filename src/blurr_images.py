# OS traversal
import os

# Importing the computer vision model
from ultralytics import YOLO

# IMG reading
import cv2

# Iteration tracking
from tqdm import tqdm

# YAML reading
import yaml

# Image data parsing
from exif import Image

# Blob storage
from azure.storage.blob import BlobServiceClient

# Hardcoding Vilnius altitude
VILNIUS_ALTITUDE = 112

# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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
    # Infering the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Creating a log.txt file to log the processed blobs
    log_file = os.path.join(current_dir, "..", "log.txt")
    log = []
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")
    else:
        log = open(log_file, "r").read().split("\n")

    # Initial class names
    class_names = None
    class_names_dict = {}

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
    blobs = [blob for blob in blobs if blob.name not in log]

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

        # Reading the exif image data
        img_metadata = Image(path)

        # Getting the altitude
        # In the default case, we would NOT blur the image if the altitude is not present
        altitude = 1000
        try:
            altitude = img_metadata.gps_altitude
        except Exception as e:  # If the altitude is not present, we will use the default
            print(f"Error reading the altitude: {e}")

        # Calculating the real height from ground
        height_from_ground = altitude - VILNIUS_ALTITUDE

        # Only blurring, if the image passes the treshold
        if height_from_ground < config["MINIMUM_HEIGHT_FROM_GROUND"]:
            print(f"Blurring {blob.name}")

            # Predicting on the downloaded image
            hat = yolo_nas.predict(
                path,
                conf=config.get("MODEL_CONFIDENCE", 0.15),
                iou=config.get("MODEL_IOU", 0.9),
            )

            # Getting the bounding boxes
            boxes = hat[0].boxes

            # Getting the labels
            labels = hat[0].boxes.cls
            labels = labels.cpu().numpy()
            label_names = [idx2class[label] for label in labels]

            # Extracting the xyxy coordinates
            bboxes = boxes.xyxy

            # Bluring
            img_cv = cv2.imread(path)

            for i, box in enumerate(bboxes):
                if label_names[i] in classes_to_blur:
                    # Getting the x, y, w, h
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

                    # Converting to int
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                    # Blurring the image
                    img_cv[y0:y1, x0:x1] = cv2.blur(
                        img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity)
                    )

            # Writing the blurred image
            cv2.imwrite(path, img_cv)

            # Uploading to the same place in the blob
            with open(path, "rb") as data:
                container_client.upload_blob(name=blob.name, data=data, overwrite=True)

        # Deleting the image from local storage
        os.remove(path)

        # Appending the blob name to the log.txt file
        with open(log_file, "a") as f:
            f.write(blob.name + "\n")


if __name__ == "__main__":
    pipeline()

# OS traversal
import os
import shutil

# Importing the NAS vision model
import super_gradients.training.models

# IMG reading
import cv2

# Iteration tracking
from tqdm import tqdm

# YAML reading
import yaml

# TMP directory creation
import tempfile

# Blob services
from azure.storage.blob import BlobServiceClient

# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Getting the sg model
yolo_nas = super_gradients.training.models.get(
    config["NAS_MODEL"], pretrained_weights="coco"
)

# Defining the classes to blurr
classes_to_blur = config["CLASSES_TO_BLURR"]

# Extracting the blurr intensity
blurr_intensity = config["BLURR_INTENSITY"]

# Extracting the input connection string and container name
input_connection_string = config["INPUT_CONNECTION_STRING"]
input_container_name = config["INPUT_CONTAINER_NAME"]

# Extracting the output connection string and container name
output_connection_string = config["OUTPUT_CONNECTION_STRING"]
output_container_name = config["OUTPUT_CONTAINER_NAME"]


def is_blob_video(blob_name: str) -> bool:
    """
    Checks if the blob is an image
    """
    return (
        blob_name.endswith(".MP4")
        or blob_name.endswith(".mp4")
        or blob_name.endswith(".AVI")
        or blob_name.endswith(".avi")
        or blob_name.endswith(".MOV")
        or blob_name.endswith(".mov")
    )


def pipeline(dl_video: bool = True) -> None:
    # Getting the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

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

    # Only leaving the videos
    blobs = [blob for blob in blobs if is_blob_video(blob.name)]

    # Creating the input directory
    input_dir = os.path.join(current_dir, "..", "input", "video")
    os.makedirs(input_dir, exist_ok=True)

    # Creating the output dir
    postprocessed_videos_dir = os.path.join(current_dir, "..", "output", "video")
    os.makedirs(postprocessed_videos_dir, exist_ok=True)

    # Downloading the videos
    if dl_video:
        for blob in tqdm(blobs, desc="Downloading the videos"):
            # Getting the base name
            base_name = os.path.basename(blob.name)

            # Getting everything except the base name from blob name
            blob_dir = blob.name.replace(base_name, "")

            # Creating the directory
            os.makedirs(os.path.join(input_dir, blob_dir), exist_ok=True)

            # Defining the path to the image
            path = os.path.join(input_dir, blob_dir, base_name)

            # Downloading the blob
            blob_client = blob_service_client.get_blob_client(
                container=input_container_name, blob=blob.name
            )
            with open(path, "wb") as f:
                f.write(blob_client.download_blob().readall())

    # Creating the containers for output
    blob_service_client = BlobServiceClient.from_connection_string(
        output_connection_string
    )

    # Getting the container
    container_client = blob_service_client.get_container_client(output_container_name)

    # Placeholder for class names
    class_names = None

    # Listing all the videps in the input dir up to the most granular level
    # Getting all the directories
    # Iterating over the directories
    # Getting all the images in the directory
    # Adding them to the list of images
    videos = []
    for root, _, files in os.walk(input_dir):
        if len(files) > 0:
            for file in files:
                if is_blob_video(file):
                    # Getting the path to the image
                    path = os.path.join(root, file)

                    # Getting the relative path
                    rel_path = os.path.relpath(path, input_dir)

                    # Adding the relative path to the list of images
                    videos.append(rel_path)

    # Iterating over the videos
    for video in videos:
        # Defining the path to video
        path = os.path.join(input_dir, video)

        # Creating a tmp dir for frames
        tmp_dir_frames = tempfile.mkdtemp()

        # Reading the video
        cap = cv2.VideoCapture(path)

        # Spliting the video by frame
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(tmp_dir_frames, f"{str(i)}.jpg"), frame)
            i += 1

        # Iterating over the frames and blurring
        # Iterating over the images
        frames = os.listdir(tmp_dir_frames)

        # Creating tmp dir for postprocessed images
        tmp_dir_postprocessed_images = tempfile.mkdtemp()

        for img in tqdm(frames, desc=f"Blurring the images for video: {video}"):
            # Defining the path to image
            path = os.path.join(tmp_dir_frames, img)

            # Reading the image with cv2
            img_cv = cv2.imread(path)

            # Predicting
            hat = yolo_nas.predict(path)

            if class_names is None:

                # Saving the list of class names
                class_names = hat[0].class_names

                # Making a dictionary where the key is the index and the value is the class name
                class_names_dict = {i: class_names[i] for i in range(len(class_names))}

            # Getting the bounding boxes
            # Extracting the predictions
            predictions = hat[0].prediction

            # Saving the bboxes
            bboxes = predictions.bboxes_xyxy

            # Saving the labels
            labels = predictions.labels

            # Getting the bounding boxes
            if len(bboxes) > 0:
                # Iterating over the boxes
                for i, box in enumerate(bboxes):
                    try:
                        # Getting the class name
                        class_name = class_names_dict[labels[i]]

                        if class_name in classes_to_blur:

                            # Getting the x, y, w, h
                            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]

                            # Converting to int
                            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                            # Blurring the image
                            img_cv[y0:y1, x0:x1] = cv2.blur(
                                img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity)
                            )

                    except Exception as e:
                        print(e)
                        continue

            # Defining the path to the image
            image_path = os.path.join(tmp_dir_postprocessed_images, img)

            # If the image exists in the postprocessed images dir, remove it
            if os.path.exists(image_path):
                os.remove(image_path)

            # Saving the image
            cv2.imwrite(image_path, img_cv)

        # Getting the base name of the video
        base_name = os.path.basename(video)

        # Getting the directory of the video
        video_dir = video.replace(base_name, "")

        # Creatign the dir
        os.makedirs(os.path.join(postprocessed_videos_dir, video_dir), exist_ok=True)

        # Listing the images
        images = os.listdir(tmp_dir_postprocessed_images)

        # Creating the image dictionary where the key is the image name and the image index
        image_dict = dict()
        for image in images:
            image_dict[image] = int(image.split(".")[0])

        # Sorting the images by the index
        images = sorted(images, key=lambda x: image_dict[x])

        # Defining the fps
        fps = 24

        # Defining the size of the video
        # Reading the first image to get the sizes
        img = cv2.imread(os.path.join(tmp_dir_postprocessed_images, images[0]))
        size = (img.shape[1], img.shape[0])

        # Defining the video writer
        out = cv2.VideoWriter(
            os.path.join(postprocessed_videos_dir, video_dir, base_name),
            cv2.VideoWriter_fourcc(*"DIVX"),
            fps,
            size,
        )

        # Iterating over the images
        for image in tqdm(images):
            # Reading the image
            img = cv2.imread(os.path.join(tmp_dir_postprocessed_images, image))

            # Writing the image
            out.write(img)

        cv2.destroyAllWindows()

        # Closing the video writer
        out.release()

        # Removing the tmp dir
        shutil.rmtree(tmp_dir_frames)
        shutil.rmtree(tmp_dir_postprocessed_images)

        # Uploading the video
        with open(
            os.path.join(postprocessed_videos_dir, video_dir, base_name), "rb"
        ) as data:
            container_client.upload_blob(name=video, data=data)


if __name__ == "__main__":
    pipeline()

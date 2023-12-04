# OS traversal 
import os 

# Importing the NAS vision model 
import super_gradients.training.models

# IMG reading 
import cv2

# Iteration tracking
from tqdm import tqdm

# YAML reading 
import yaml

# Blob storage
from azure.storage.blob import BlobServiceClient

# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Getting the sg model 
yolo_nas = super_gradients.training.models.get(config['NAS_MODEL'], pretrained_weights="coco")

# Defining the classes to blurr
classes_to_blur = config['CLASSES_TO_BLURR']

# Extracting the blurr intensity 
blurr_intensity = int(config['BLURR_INTENSITY'])

# Extracting the input connection string and container name 
input_connection_string = config['INPUT_CONNECTION_STRING']
input_container_name = config['INPUT_CONTAINER_NAME']

# Extracting the output connection string and container name
output_connection_string = config['OUTPUT_CONNECTION_STRING']
output_container_name = config['OUTPUT_CONTAINER_NAME']

def is_blob_image(blob_name: str) -> bool:
    """
    Checks if the blob is an image
    """
    return blob_name.endswith('.jpg') or blob_name.endswith('.png') or blob_name.endswith('.jpeg') or blob_name.endswith('.JPG') or blob_name.endswith('.PNG') or blob_name.endswith('.JPEG')

# Defining the pipeline 
def pipeline() -> None:
    # Infering the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Initial class names 
    class_names = None 
    class_names_dict = {}

    # Creating the connection 
    blob_service_client = BlobServiceClient.from_connection_string(input_connection_string)

    # Getting the container
    container_client = blob_service_client.get_container_client(input_container_name)

    # Listing all the blobs 
    blobs = container_client.list_blobs()

    # Only leaving the images
    blobs = [blob for blob in blobs if is_blob_image(blob.name)]

    # Creating the input directory 
    input_dir = os.path.join(current_dir, '..', 'input', "images")
    os.makedirs(input_dir, exist_ok=True)

    # Creating the output dir 
    postprocessed_images_dir = os.path.join(current_dir, '..', 'output', "images")
    os.makedirs(postprocessed_images_dir, exist_ok=True)

    # Downloading the images
    for blob in tqdm(blobs, desc='Downloading the images'): 
        # Getting the base name
        base_name = os.path.basename(blob.name)

        # Getting everything except the base name from blob name 
        blob_dir = blob.name.replace(base_name, '')

        # Creating the directory
        os.makedirs(os.path.join(input_dir, blob_dir), exist_ok=True)

        # Defining the path to the image 
        path = os.path.join(input_dir, blob_dir, base_name)

        # Downloading the blob 
        blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=blob.name)
        with open(path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    # Listing all the images in the input dir up to the most granular level
    # Getting all the directories
    # Iterating over the directories
    # Getting all the images in the directory
    # Adding them to the list of images
    images = []
    for root, _, files in os.walk(input_dir):
        if len(files) > 0:
            for file in files:
                if is_blob_image(file):
                    # Getting the path to the image
                    path = os.path.join(root, file)

                    # Getting the relative path
                    rel_path = os.path.relpath(path, input_dir)

                    # Adding the relative path to the list of images
                    images.append(rel_path)   

    # Iterating over the images 
    for img in tqdm(images, desc='Blurring the images'): 

        # Reading the image with cv2
        img_cv = cv2.imread(os.path.join(input_dir, img))

        # Predicting 
        hat = yolo_nas.predict(os.path.join(input_dir, img))

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
                        img_cv[y0:y1, x0:x1] = cv2.blur(img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity))

                except Exception as e:
                    print(e)
                    continue

        # Getting the base name of the image 
        base_name = os.path.basename(img)

        # Getting everything except the base name from blob name
        blob_dir = img.replace(base_name, '')

        # Creating the output directory 
        output_dir = os.path.join(postprocessed_images_dir, blob_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Defining the full path to the image
        image_path = os.path.join(output_dir, base_name)

        # Saving the image
        cv2.imwrite(image_path, img_cv)

    # Uploading the blurred images to the output container
    # Creating the connection
    blob_service_client = BlobServiceClient.from_connection_string(output_connection_string)

    # Getting the container
    container_client = blob_service_client.get_container_client(output_container_name)

    # Iterating over the output images and uploading 
    for root, _, files in os.walk(postprocessed_images_dir):
        if len(files) > 0:
            for file in tqdm(files, desc=f'Uploading the images from {postprocessed_images_dir} to {output_container_name}'):
                # Defining the path to the image
                path = os.path.join(root, file)

                # Getting the relative path
                rel_path = os.path.relpath(path, postprocessed_images_dir)

                # Uploading the blob
                with open(path, "rb") as data:
                    container_client.upload_blob(name=rel_path, data=data)

if __name__ == '__main__': 
    pipeline()
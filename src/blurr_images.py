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

# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Getting the sg model 
yolo_nas = super_gradients.training.models.get(config['NAS_MODEL'], pretrained_weights="coco")

# Defining the classes to blurr
classes_to_blur = config['CLASSES_TO_BLURR']

# Extracting the blurr intensity 
blurr_intensity = int(config['BLURR_INTENSITY'])

# Defining the pipeline 
def pipeline() -> None:
    # Infering the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Initial class names 
    class_names = None 
    class_names_dict = {}

    # Defining the dir for the postprocessed images
    postprocessed_images_dir = os.path.join(current_dir, '..', 'output', "postprocessed_images")
    if not os.path.exists(postprocessed_images_dir):
        os.makedirs(postprocessed_images_dir, exist_ok=True)

    # Listing the images/
    images_dir = os.path.join(current_dir, '..', 'input', "images")
    images = os.listdir(images_dir)

    # Iterating over the images 
    for img in tqdm(images, desc='Blurring the images'): 
        # Defining the path to image
        path = os.path.join(images_dir, img)

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
                        img_cv[y0:y1, x0:x1] = cv2.blur(img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity))

                except Exception as e:
                    print(e)
                    continue

        # Defining the path to the image
        image_path = os.path.join(postprocessed_images_dir, img)

        # If the image exists in the postprocessed images dir, remove it
        if os.path.exists(image_path):
            os.remove(image_path)

        # Saving the image
        cv2.imwrite(image_path, img_cv)

if __name__ == '__main__': 
    pipeline()
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

# TMP directory creation 
import tempfile

# Reading the configuration file
with open("configuration.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Getting the sg model 
yolo_nas = super_gradients.training.models.get(config['NAS_MODEL'], pretrained_weights="coco")

# Defining the classes to blurr
classes_to_blur = config['CLASSES_TO_BLURR']

# Extracting the blurr intensity 
blurr_intensity = config['BLURR_INTENSITY']

def pipeline() -> None: 
    # Getting the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Path to videos 
    videos_dir = os.path.join(current_dir, '..', 'input', "videos")
    videos_dir_output = os.path.join(current_dir, '..', 'output', "postprocessed_videos")
    if not os.path.exists(videos_dir_output):
        os.makedirs(videos_dir_output, exist_ok=True)

    # Listing all the videos
    videos = os.listdir(videos_dir)

    # Placeholder for class names 
    class_names = None

    # Iterating over the videos
    for video in videos:
        # Defining the path to video
        path = os.path.join(videos_dir, video)

        # Creating a tmp dir for frames
        tmp_dir_frames = tempfile.mkdtemp()

        # Reading the video 
        cap = cv2.VideoCapture(path)

        # Spliting the video by frame 
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(tmp_dir_frames, f'{str(i)}.jpg'), frame)
            i+=1

        # Iterating over the frames and blurring
        # Iterating over the images 
        frames = os.listdir(tmp_dir_frames)

        # Creating tmp dir for postprocessed images
        tmp_dir_postprocessed_images = tempfile.mkdtemp()

        for img in tqdm(frames, desc=f'Blurring the images for video: {video}'): 
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
                            img_cv[y0:y1, x0:x1] = cv2.blur(img_cv[y0:y1, x0:x1], (blurr_intensity, blurr_intensity))

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

        # Defining the path to the video
        video_path = os.path.join(videos_dir_output, video)

        images = os.listdir(tmp_dir_postprocessed_images)

        # Creating the image dictionary where the key is the image name and the image index 
        image_dict = dict()
        for image in images:
            image_dict[image] = int(image.split('.')[0])

        # Sorting the images by the index
        images = sorted(images, key=lambda x: image_dict[x])

        # Defining the fps
        fps = 24

        # Defining the size of the video
        # Reading the first image to get the sizes 
        img = cv2.imread(os.path.join(tmp_dir_postprocessed_images, images[0]))
        size = (img.shape[1], img.shape[0])

        # Defining the video writer
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

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
        os.remove(tmp_dir_frames)
        os.remove(tmp_dir_postprocessed_images)

if __name__ == '__main__':
    pipeline()
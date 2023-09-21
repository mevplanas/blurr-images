# Blurr images

Repository that blurrs certain objects from videos. 

# Models used

## WALDO 

The waldo model is downloaded from [here](https://github.com/stephansturges/WALDO), along with an in depth documentation. 

## YOLO NAS model 

The YOLO NAS model is taken from the super gradients package [here](https://pypi.org/project/super-gradients/). 

# Blurring the videos 

The structure of the repository is as follows: 

```
├───configuration.yml
├───blurr_video_sg.ipynb
├───blurr_video_waldo.ipynb
├───example_image
├───extracted_images/
├───models/
├───output/
├───postprocessed_images/
└───videos/
```

## Configuration file 

```yaml
VIDEO_TO_USE: DJI_0154.MP4

# WALDO model
MODEL_TO_USE_WALDO: yolov7-W25-1088px-newDefaults-bs96-best-topk-200.onnx 

# Steps to do 
STEPS:
  SPLIT_VIDEO: True

# Defining the classes available for WALDO model 
WALDO_CLASSES: 
  - car 
  - van
  - truck
  - building 
  - human
  - gastank
  - digger
  - container
  - bus
  - u_pole
  - boat
  - bike
  - smoke
  - solarpanels
  - arm
  - plane

# Defining the classes to plot 
WALDO_CLASSES_PLOT:
  - car
  - van 
  - truck
  - human
  - bus
  - bike

# Defining the waldo classes to blurr 
WALDO_CLASSES_BLURR:
  - human

# Defining the number of frames to plot 
FRAMES_TO_PLOT: 500
```

# Running the notebooks

Currently, there are two notebooks: 

* blurr_video_sg.ipynb - blurs the videos using the YOLO NAS model
* blurr_video_waldo.ipynb - blurs the videos using the WALDO model

The workflow is as follows:

* A user puts a video in the videos/ directory. 
* The notebooks create directories for the videos in the extracted_images/ and postprocessed_images/ directories.
* Finally, an output video is created in the output/ directory.


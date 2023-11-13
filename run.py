# Importing the pipelines from src directory 
from src.blurr_images import pipeline as blurr_images_pipeline

# Running the pipeline
if __name__ == '__main__':
    blurr_images_pipeline()
# Blurr images

Repository that blurrs certain objects from videos. 

The main commands are: 

```
python -m src.blurr_images
```

```
python -m src.blurr_videos
```

# Configuration file 

The configuration for the script running is in the `configuration.yml` file. 

# Virtual environment 

## Virtualenv

The needed packages are stored in the requirements.txt file.

To create virtual env with python 3.11, run the following command: 

```bash
virtualenv blurr_env --python=python3.11
```

To activate it, run the following command: 

```bash
source blurr_env/bin/activate
```

On windows, run: 

```bash
blurr_env\Scripts\activate.bat
```

## Anaconda 

To install the env using the `env.yml` file, run the command:

```bash
conda env create -f env.yml
```

To update the env, run the command: 

```bash
conda env update --file env.yml --prune
```

To activate the env, run the command: 

```bash
conda activate blurr_env
```

## Docker 

To create a complete Linux environment, build the image using the following command: 

```bash
docker build -t blurr .
```

To run the container linking the `input` and `output` directories to the `app` directory, run the following command: 

(Bash)
```bash
docker run -it --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output blurr
```

(Powershell)
```bash
docker run -it --rm -v ${pwd}/input:/app/input -v ${pwd}/output:/app/output blurr
```

# YOLO NAS model 

The YOLO NAS model is taken from the super gradients package [here](https://pypi.org/project/super-gradients/). 

The model is used to find certain classes and blurr them from images.
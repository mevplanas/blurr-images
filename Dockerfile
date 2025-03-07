# Base python 3.11 image
FROM python:3.11-slim-buster

# Setting to non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Updating the apt-get package manager
RUN apt-get update

# Installing gcc
RUN apt-get install -y gcc

# Install ffmpeg and other dependencies
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Copying over the requirements.txt file
COPY requirements.txt requirements.txt

# Installing the requirements
RUN pip3 install -r requirements.txt

# Creating the app directory
WORKDIR /app

# Copying over the main scripts
COPY src src

# Main run script
COPY run.py run.py

# Copying over the configuration.yml file
COPY configuration.yml configuration.yml

# The command to run the script
CMD ["python3", "run.py"]

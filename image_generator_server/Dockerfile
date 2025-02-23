# Use an official CUDA runtime and development image as a parent image
#FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container to /image-caption-extractor
WORKDIR /image_generator_server

# Install Python
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools wheel
    
# Add the requirements file first to leverage Docker cache
COPY image_generator_server/requirements.txt .
    
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY image_generator_server/app/ /image_generator_server/app/
COPY image_generator_server/demo_client.py .

WORKDIR /image_generator_server/app

# Make port 6070 available to the world outside this container
EXPOSE 6070

# Define the command to run the application
CMD ["python3", "main.py"]
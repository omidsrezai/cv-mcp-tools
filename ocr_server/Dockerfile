# Use an official CUDA runtime and development image as a parent image
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /ocr_server

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*  # Clean up APT cache to reduce image size

# Ensure `python3` is the default Python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first to leverage Docker cache
COPY ocr_server/requirements.txt requirements.txt

# Install PyTorch first to avoid `flash-attn` failure
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0

# Now install the rest of the dependencies (excluding torch & torchvision)
RUN grep -Ev "torch|torchvision" requirements.txt > temp.txt && mv temp.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install Janus without removing other dependencies
RUN git clone https://github.com/deepseek-ai/Janus.git /ocr_server/Janus && \
    pip install -e /ocr_server/Janus

# Copy application code
COPY ocr_server/app /ocr_server/app
COPY ocr_server/demo_client.py /ocr_server/demo_client.py
COPY ocr_server/assets /ocr_server/assets

# Set the working directory to the application directory
WORKDIR /ocr_server/app

# Expose necessary ports
EXPOSE 6080 6081

# Define the command to run the application
CMD ["python", "main.py"]
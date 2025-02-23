# Image Generator (FLUX.1-Schnell) Server
 
- A Gradio service for image generation using [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell). 
- By default, the service runs on port 6070, but this can be changed via the PORT environment variable.

## Directory Structure
- `app/`: Contains the main application logic.
- `demo_client.py`: A sample client script to demonstrate how to interact with the service.
- `Dockerfile`: Defines the Docker environment for deploying the service.
- `requirements.txt`: Specifies the Python packages required for the application.

## Docker

### Building Image:
```shell
docker buildx build -t flux-schnell -f image_generator_server/Dockerfile .
```

### Running Container:
```shell
docker run --gpus all \
  -e PORT=<port_num> \  # Optional: Default is 6070
  -p 6070:6070 \
  --name flux-schnell flux-schnell:latest
```

## Demo
After running the Docker container, open a separate terminal and run:
```shell
python demo_client.py
```
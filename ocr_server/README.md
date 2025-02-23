# OCR Server

- A Gradio service for OCR and image understanding using [Qwen-VL](https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct) and [Janus](https://huggingface.co/deepseek-ai/Janus-Pro-7B) models.
- By default, the service runs on ports 6080 (UI) and 6081 (API), but these can be changed via the PORT_UI and PORT_API environment variables.

## Directory Structure
- `app/`: Contains the main application logic
- `requirements.txt`: Specifies the Python packages required for the application

## Environment Variables
- `PORT_UI`: UI interface port (default: 6080)
- `PORT_API`: API interface port (default: 6081)
- `MODEL_TYPE`: Default model to use ("Qwen-VL" or "Janus", default: "Qwen-VL")

## Docker

### Building Image:
```shell
docker buildx build -t ocr-server -f Dockerfile .
```

### Running Container:
```shell
docker run --gpus all \
  -e PORT_UI=6080 \  # Optional: Default is 6080
  -e PORT_API=6081 \  # Optional: Default is 6081
  -e MODEL_TYPE=Qwen-VL \  # Optional: Default model
  -p 6080:6080 \
  -p 6081:6081 \
  --name ocr-server ocr-server:latest
```

## Usage

### UI Interface
After running the Docker container, visit `http://localhost:6080` to access the web interface where you can:
- Upload images
- Enter custom prompts
- Choose between Qwen-VL and Janus models
- Adjust generation parameters

## Demo
After running the Docker container, open a separate terminal and run:
```shell
python demo_client.py
```

### API Interface
The service also provides a REST API endpoint at `http://localhost:6081` that accepts POST requests with JSON payload:

```json
{
    "image": "base64_encoded_image_string",
    "prompt": "Please extract all text from this image",
    "max_tokens": 256,
    "model_type": "Qwen-VL"
}
```
see `demo_client.py` for an examlple.
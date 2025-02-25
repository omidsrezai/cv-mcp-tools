import asyncio
import base64
import io
import os
import time  
from typing import Union, Tuple

import httpx
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

# Define the image generation API endpoint
IMAGEN_API_ENDPOINT = os.getenv("IMAGEN_API_ENDPOINT", "https://localhost:6070/api/predict")

# Define the OCR API endpoint
OCR_API_ENDPOINT = os.getenv("OCR_API_ENDPOINT", "http://localhost:6081/gradio_api/api/predict")

# MinIO Configuration
MINIO_URL = os.getenv("MINIO_URL", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Initialize FastMCP server
mcp = FastMCP("ocr_imagen", max_buffer_size=500)

# Initialize MinIO client
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if you're using HTTPS
)

# # Define a prompt template
# @mcp.prompt()
# def summarize_text(text: str) -> str:
#     """Prompt to summarize the provided text."""
#     return f"Please provide a concise summary of the following text:\n\n{text}"


@mcp.prompt()
def iterative_image_generation_validation():
    return f"""Iterative Image Generation Validation Process
    This prompt structures the process into clear, iterative steps to generate an image, validate it with OCR, and retry if necessary. 
    You are a technician who must follow this process line by line exactly. You are not allowed to stop or ask the user any questions 
    until you complete the algorithm. 

    Remember:
    - You must complete **10 attempts**, starting at attempt **1**.
    - After each attempt, you must report the values of `attempt`, `seed`, and `num_steps` to the user.

    Step A: Generate Initial Image
    - Use the `text_to_image_generator()` tool with the following parameters:
    - `bucket_name`: `"generate-images-tmp"`
    - `object_name`: `[base_name]_[attempt].png` (e.g., `blackboard_1.png`)
    - `prompt`: **Text description for image generation**
    - `seed`: Start at **42**
    - `num_steps`: Start at **5** (valid range: **5-10**)

    Step B: Validate with OCR
    - Use the `ocr_extractor()` tool with the same bucket and object name.
    - Compare the extracted text with the original prompt.

    Step C: Optimization Loop
    1. **Check attempt count:**
    - If `attempt >= 10`, proceed to **Step D**.
    
    2. **Check OCR validation:**
    - If the extracted text matches the prompt, proceed to **Step D**.
    - If the extracted text does not match the prompt:
        - **Increment `attempt` by 1** (`attempt = attempt + 1`).
        - Adjust parameters as follows:
        1. If `num_steps < 10`, increment `num_steps` (`num_steps = num_steps + 1`) while keeping `seed` unchanged.
        2. If `num_steps ≥ 10`, reset `num_steps` to **5** and increment `seed` (`seed = seed + 1`).
        - Update `object_name` with the new attempt number.
        - **Repeat Steps A-B.**
    
    Step D: Finalize
    - **If the last attempt has a perfect text match, use that image.**
    - **If no perfect match is found, select the best image with the closest match.**
    - Perform the following actions:
    1. **Copy the selected image to the final location:**
        - Use the `copy_image_between_buckets()` tool with:
        - `target_bucket`: `"generate-images-final"`
        - `target_object`: Remove the attempt number (e.g., `blackboard_5.png → blackboard.png`).
    2. **Retrieve the final image using the `get_image_from_minio()` tool.**
    """

@mcp.tool()
async def text_to_image_generator(bucket_name: str, object_name: str, prompt: str, seed: int,
                         num_steps: int = 25, height: int = 960, width: int = 544,
                         max_seq_length: int = 512, return_image: bool = False) -> Union[bool, Tuple[bool, Image]]:
    """
    Generates an image using Flux Schnell image generation server and uploads it to a MinIO bucket.

    Args:
        bucket_name (str): Name of the MinIO bucket where the image will be stored.
        object_name (str): Name to use for the uploaded image in the bucket.
        prompt (str): The text prompt for image generation.
        seed (int): Seed value for reproducibility.
        num_steps (int, optional): Number of inference steps. Default is 25.
        height (int, optional): Image height (must be divisible by 8). Default is 960.
        width (int, optional): Image width (must be divisible by 8). Default is 544.
        max_seq_length (int, optional): Maximum sequence length. Default is 512.
        return_image (bool, optional): Whether to return the generated image.

    Returns:
        Union[bool, Tuple[bool, Image]]: 
            - If `return_image` is False, returns a boolean indicating success.
            - If `return_image` is True, returns a tuple containing:
              1. A boolean indicating success.
              2. A `fastmcp.Image` object of the generated image.
    """
    try:
        # Step 1: Prepare the payload
        payload = {
            "data": [
                prompt,        # Text prompt for image generation
                seed,          # Seed value for reproducibility
                num_steps,     # Number of inference steps
                height,        # Image height (should be divisible by 8)
                width,         # Image width (should be divisible by 8)
                max_seq_length # Max sequence length
            ]
        }

        # Step 2: Send request to image generation API (async)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(IMAGEN_API_ENDPOINT, json=payload)

        # Step 3: Process API response
        if response.status_code != 200:
            raise ValueError(f"Failed to get a response from the image generation API, status code: {response.status_code}")

        result = response.json()

        if "data" not in result or len(result["data"]) < 2:
            raise ValueError(f"Error in image generation response: {result.get('error', 'Unknown error occurred')}")

        status, image_info = result["data"]
        
        if not image_info or "url" not in image_info:
            raise ValueError(f"Image generation failed: {status}")

        image_url = image_info["url"]

        # Step 4: Download the generated image (async)
        async with httpx.AsyncClient(timeout=30) as client:
            image_response = await client.get(image_url)

        if image_response.status_code != 200:
            raise ValueError(f"Failed to download generated image, status code: {image_response.status_code}")

        image_data = image_response.content  # Get image bytes

        # Step 5: Upload the image to MinIO (async handling)
        bucket_exists = await asyncio.to_thread(minio_client.bucket_exists, bucket_name)
        if not bucket_exists:
            await asyncio.to_thread(minio_client.make_bucket, bucket_name)

        # Convert image data bytes to a stream
        image_data_bytes_length = len(image_data)
        image_data_stream = io.BytesIO(image_data)

        try:
            await asyncio.to_thread(
                minio_client.put_object, bucket_name, object_name, image_data_stream, image_data_bytes_length
            )
        except S3Error as e:
            raise ValueError(f"Error uploading image to MinIO: {e}")

        # Step 6: Return success status (and optionally, the image)
        if return_image:
            return True, Image(data=image_data, format="png")  # Using FastMCP's Image
        else:
            return True  # Only return success flag

    except S3Error as e:
        raise ValueError(f"Error uploading image to MinIO: {e}")
    except Exception as e:
        raise ValueError(f"Error in generate_image: {e}")


@mcp.tool()
async def ocr_extractor(bucket_name: str, object_name: str, prompt: str ="Please extract all text from this image.", return_annotated_image: bool = False, model_type: str = "Qwen-VL") -> Union[str, Tuple[str, Image]]:
    """
    Performs OCR on an image stored in a MinIO bucket using a Gradio-based OCR API.

    Args:
        bucket_name (str): Name of the MinIO bucket containing the image.
        object_name (str): Path of the image object in the MinIO bucket.
        prompt (str): The text prompt to guide the OCR process.
        return_annotated_image (bool): Whether to return the annotated image.
        model_type (str, optional): The model to use for OCR. Options are "Qwen-VL" (default) or "Janus".

    Returns:
        Union[str, Tuple[str, Image]]: 
            - If `return_annotated_image` is False, returns the extracted text.
            - If `return_annotated_image` is True, returns a tuple containing:
              1. The extracted text.
              2. A `fastmcp.Image` object representing the annotated image.
    """
    max_retries = 3  # Number of retries if the server is unresponsive
    retry_delay = 5  # Seconds to wait before retrying

    for attempt in range(max_retries):
            try:
                # Validate model_type
                if model_type not in {"Qwen-VL", "Janus"}:
                    raise ValueError(f"Invalid model_type '{model_type}'. Supported options: 'Qwen-VL', 'Janus'.")

                # Step 1: Retrieve the image from MinIO (async)
                response = await asyncio.to_thread(minio_client.get_object, bucket_name, object_name)
                image_data = response.read()
                response.close()
                response.release_conn()

                # Step 2: Convert the image data to base64
                image_base64 = base64.b64encode(image_data).decode("utf-8")

                # Step 3: Prepare the payload
                payload = {
                    "data": [{
                        "image": image_base64,  # Base64 encoded image
                        "prompt": prompt,       # Dynamic prompt input
                        "max_tokens": 128,      # Maximum number of tokens for response
                        "model_type": model_type  # Model selection parameter
                    }]
                }

                # Step 4: Make the API request (async)
                async with httpx.AsyncClient(timeout=30) as client:
                    api_response = await client.post(OCR_API_ENDPOINT, json=payload)

                # Step 5: Process the API response
                if api_response.status_code != 200:
                    raise ValueError(f"OCR request failed, status code: {api_response.status_code}")

                result = api_response.json()

                if "data" not in result or not result["data"]:
                    raise ValueError(f"OCR API error: {result.get('error', 'Unknown error occurred')}")

                extracted_text = result["data"][0]["text"]

                # Step 6: Return results
                if return_annotated_image:
                    return extracted_text, Image(data=image_data, format="png")  # Using FastMCP Image
                else:
                    return extracted_text

            except (httpx.TimeoutException, httpx.ConnectError):
                print(f"Attempt {attempt + 1}: OCR server is not responding. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)  # Use async sleep instead of time.sleep
            except S3Error as e:
                raise ValueError(f"Error retrieving image from MinIO: {e}")
            except Exception as e:
                raise ValueError(f"Error in OCR function: {e}")

    print("OCR server is not responding after multiple attempts. OCR request failed.")
    return "OCR request failed due to server timeout or unavailability."


@mcp.tool()
async def get_image_from_minio(bucket_name: str, object_name: str) -> Image:
    """
    Retrieves an image from a specified MinIO bucket and returns it as a FastMCP Image.

    Args:
        bucket_name (str): Name of the MinIO bucket containing the image.
        object_name (str): Name of the image object in the MinIO bucket.

    Returns:
        Image: A FastMCP Image object containing the image data.
    """
    try:
        # Get the object from MinIO as a stream
        response = minio_client.get_object(
            bucket_name,
            object_name
        )
        
        # Read the image bytes from the stream
        image_data = response.read()
        
        # Ensure the stream is closed
        response.close()
        response.release_conn()

        # Return the image as a FastMCP Image object
        return Image(data=image_data, format="png")  # Assuming the image format is PNG
    except S3Error as e:
        raise ValueError(f"Error retrieving image from MinIO bucket '{bucket_name}': {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

@mcp.tool()
async def put_image_to_minio(bucket_name: str, object_name: str, image_path: str) -> str:
    """
    Uploads an image to a MinIO bucket. If the bucket doesn't exist, it will create the bucket.

    Args:
        bucket_name (str): Name of the MinIO bucket.
        object_name (str): Name to use for the uploaded image in the bucket.
        image_path (str): Local file path of the image to upload.

    Returns:
        str: Confirmation message indicating successful upload.
    """
    try:
        # Check if the bucket exists, and create it if it doesn't
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created.")

        # Upload the image to the specified bucket
        minio_client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=image_path
        )
        return f"Image '{object_name}' uploaded successfully to bucket '{bucket_name}'."
    except S3Error as e:
        raise ValueError(f"Error uploading image to MinIO bucket '{bucket_name}': {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

@mcp.tool()
async def copy_image_between_buckets(source_bucket: str, source_object: str, target_bucket: str, target_object: str) -> str:
    """
    Copies an image from one MinIO bucket to another.

    Args:
        source_bucket (str): Name of the source MinIO bucket.
        source_object (str): Name of the object in the source bucket.
        target_bucket (str): Name of the target MinIO bucket.
        target_object (str): Name for the copied object in the target bucket.

    Returns:
        str: Confirmation message indicating successful copy.
    """
    try:
        # Check if the source object exists
        stat = minio_client.stat_object(source_bucket, source_object)

        # Check if the target bucket exists, and create it if it doesn't
        if not minio_client.bucket_exists(target_bucket):
            minio_client.make_bucket(target_bucket)
            print(f"Bucket '{target_bucket}' created.")

        # Copy the object from source bucket to target bucket
        #copy_source = f"/{source_bucket}/{source_object}"
        minio_client.copy_object(
            bucket_name=target_bucket,
            object_name=target_object,
            source=CopySource(source_bucket, source_object)
        )
        return f"Image '{source_object}' successfully copied from bucket '{source_bucket}' to '{target_bucket}' as '{target_object}'."
    except S3Error as e:
        raise ValueError(f"Error copying image: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

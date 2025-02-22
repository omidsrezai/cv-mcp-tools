import os
from typing import Union, Tuple

from ultralytics import YOLO
from mcp.server.fastmcp import FastMCP, Image
import io
from PIL import Image as PILImage

from minio import Minio
from minio.error import S3Error


# MinIO Configuration
MINIO_URL = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# YOLO model to use
YOLO_MODEL = "yolo11m.pt"

# Initialize FastMCP server
mcp = FastMCP("object_detector")

# Initialize MinIO client
minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Set to True if you're using HTTPS
)

@mcp.tool()
async def save_file_to_folder(folder_path: str, file_name: str, file_data: bytes) -> str:
    """
    Saves a file to a specified folder. If the folder doesn't exist, it creates the folder.

    Args:
        folder_path (str): Path to the folder where the file will be saved.
        file_name (str): Name to use for the saved file.
        file_data (bytes): Binary data of the file to save.

    Returns:
        str: Full path of the saved file.

    Raises:
        ValueError: If there is an error saving the file.
    """
    try:
        # Ensure the folder exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")

        # Define the full file path
        full_file_path = os.path.join(folder_path, file_name)

        # Write the file data to the specified path
        with open(full_file_path, "wb") as file:
            file.write(file_data)

        return f"File '{file_name}' saved successfully to folder '{folder_path}'. Full path: {full_file_path}"
    except Exception as e:
        raise ValueError(f"Error saving file to folder '{folder_path}': {e}")
    
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
async def list_detectable_categories() -> str:
    """Get the list of all object categories that this server can detect.
    """
    # Load a model
    model = YOLO(YOLO_MODEL)  # pretrained YOLO11n model

    return ", ".join([f"{class_name}" for class_name in model.names.values()])

@mcp.tool()
async def detect_object(bucket_name: str, object_name: str, return_annotated_image: bool) -> Union[str, Tuple[str, Image]]:
    """
    Identify object categories and count their occurrences as defined by the MS COCO dataset. 
    The complete list of recognizable objects can be accessed using model.names().

    Args:
        bucket_name (str): Name of the MinIO bucket containing the image.
        object_name (str): Path of the image object in the MinIO bucket.
        return_annotated_image (bool): Whether to return an annotated image.

    Returns:
        Union[str, Tuple[str, Image]]: 
            - If `return_annotated_image` is False, returns a string describing object counts.
            - If `return_annotated_image` is True, returns a tuple containing:
              1. A string describing object counts.
              2. A `fastmcp.Image` object representing the annotated image.
    """
    try:
        # Step 1: Retrieve the image from MinIO
        response = minio_client.get_object(bucket_name, object_name)
        image_data = response.read()  # Read the image bytes from the stream
        response.close()
        response.release_conn()

        # Save the image data to a temporary file for YOLO processing
        temp_image_path = "/tmp/temp_image.jpg"  # You can use tempfile for dynamic file names
        with open(temp_image_path, "wb") as f:
            f.write(image_data)

        # Step 2: Load the YOLO model
        model = YOLO(YOLO_MODEL)  # Pretrained YOLO model

        # Step 3: Perform inference
        results = model(temp_image_path, conf=0.45)  # Returns a list of Results objects
        if not results:
            raise ValueError("No results returned from the model.")
        
        result = results[0]  # We assume there's one result per image

        # Step 4: Extract class IDs and count occurrences
        cls_list = result.boxes.cls.int().tolist()
        class_counts = {}
        for cls_id in cls_list:
            class_name = result.names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Generate textual result
        text_result = ", ".join([f"{class_name}: {count}" for class_name, count in class_counts.items()])
        
        # Step 5: Return annotated image if requested
        if return_annotated_image:
            annotated_image_bgr = result.plot()  # This returns the image in BGR format

            # Convert BGR to RGB
            annotated_image_rgb = annotated_image_bgr[..., ::-1]

            # Convert the NumPy array to a Pillow image
            rgb_image = PILImage.fromarray(annotated_image_rgb, mode="RGB")

            # Resize the image so the largest dimension is 100 pixels
            rgb_image.thumbnail((600, 600))
            
            # Save the image to an in-memory buffer
            buffer = io.BytesIO()
            rgb_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Return both the textual result and the annotated image
            return text_result, Image(data=buffer.read(), format="png")
        else:
            # Return only the textual result
            return text_result
    except S3Error as e:
        raise ValueError(f"Error retrieving image from MinIO: {e}")
    except Exception as e:
        raise ValueError(f"Error in detect_object: {e}")
    
    

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')


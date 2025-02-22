import requests
from PIL import Image
from io import BytesIO
import os

# Define the API endpoint
url = "http://localhost:6070/api/predict"

# Define the payload with the parameters
prompt = "A sunset over the mountain"
seed = 42
num_steps = 10
height = 960
width = 544
max_seq_length = 512

payload = {
    "data": [
        prompt,       # The prompt
        seed,         # Seed value
        num_steps,    # Number of inference steps
        height,       # Image height (divisible by 8)
        width,        # Image width (divisible by 8)
        max_seq_length  # Max sequence length
    ]
}

# Make the POST request to the API
response = requests.post(url, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()

    if "data" in result and len(result["data"]) == 2:
        status = result["data"][0]
        image_info = result["data"][1]
        
        # Check if image info is valid
        if image_info and "url" in image_info:
            image_url = image_info["url"]
            try:
                # Download the image from the URL
                image_response = requests.get(image_url)
                image = Image.open(BytesIO(image_response.content))
                #image = Image.open(BytesIO(requests.get(image_url).content))
                
                # Ensure the 'images' folder exists
                if not os.path.exists("images"):
                    os.makedirs("images")
                
                # Create a filename based on the first three words of the prompt, seed, and num_steps
                prompt_words = "_".join(prompt.split()[:3])
                filename = f"{prompt_words}_{seed}_{num_steps}.jpg"
                save_path = os.path.join("images", filename)
                
                # Save the image as a JPG file in the 'images' folder
                image.save(save_path, "JPEG")
                
                print(f"Image saved to {save_path}")
            except Exception as e:
                print(f"An error occurred while processing the image: {e}")
        else:
            # Handle the case where the response contains an error message
            print(f"Error: {status}")
    else:
        # If the response does not match the expected format, assume it's an error
        print(f"Error: {result.get('error', 'Unknown error occurred')}")
else:
    # Failed to get a response from the server
    print(f"Failed to get a response, status code: {response.status_code}")
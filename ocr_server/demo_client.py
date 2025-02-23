import base64
import requests


# Define the API endpoint
url = "http://localhost:6081/gradio_api/api/predict"

# Load an image and convert it to base64
image_path = "assets/1.png"  # Replace with your image path
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Define the payload with the parameters
prompt = "Describe this image"
max_tokens = 128

my_load = {
        "image": image_base64,  # Base64 encoded image
        "prompt": prompt,        # Text prompt
        "max_tokens": max_tokens,     # Maximum number of tokens for response
        "model_type": "Janus"
}

payload = {
    "data": [my_load]
}

# payload = {
#     "data": [
#         image_base64,  # Base64 encoded image
#         prompt,        # Text prompt
#         max_tokens     # Maximum number of tokens for response
#     ]
# }

# Make the POST request to the API
response = requests.post(url, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()

    if "data" in result:

        generated_text = result["data"][0]["text"]
        print(f"Generated Text: {generated_text}")
    else:
        print(f"Error: {result.get('error', 'Unknown error occurred')}")
else:
    print(f"Failed to get a response, status code: {response.status_code}")
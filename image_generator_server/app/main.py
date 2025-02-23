import base64
from io import BytesIO
import os
import queue
import threading

from diffusers import FluxPipeline
import gradio as gr
from PIL import Image
import torch


default_port = int(os.getenv('PORT', 6070))

DEFAULT_IMAGE_HEIGHT = 960
DEFAULT_IMAGE_WIDTH = 544
DEFAULT_MAX_SEQ_LENGTH = 512


class Flux_Schnell:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()
        self.result = None

    def run_image_generation(self, prompt, seed, num_steps=4, height=960, width=544, max_seq_length=512):
        with self.lock:
            image = self.pipe(
                prompt,
                guidance_scale=0,
                output_type="pil",
                height=height,
                width=width,
                num_inference_steps=num_steps,
                max_sequence_length=max_seq_length,
                generator=torch.Generator(self.device).manual_seed(seed)
            ).images[0]
            return image
            
    def process_queue(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            prompt, seed, num_steps, height, width, max_seq_length, callback = task
            try:
                result = self.run_image_generation(prompt, seed, num_steps, height, width, max_seq_length)
                callback(result)
            except Exception as e:
                callback({"error": str(e)})
            finally:
                self.task_queue.task_done()

    def validate_dimensions(self, height, width):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and Width must be divisible by 8.")

    def generate_image(self, prompt, seed, num_steps, height, width, max_seq_length):
        try:
            # Validate height and width
            self.validate_dimensions(height, width)
            
            # Check if the task queue is not empty
            if not self.task_queue.empty():
                return "Task queue is busy, please try again later.", None

            def process_result(result):
                self.result = result

            # Put the task in the queue, including height, width, and max_seq_length
            self.task_queue.put((prompt, seed, num_steps, height, width, max_seq_length, process_result))
            self.task_queue.join()

            if isinstance(self.result, dict) and "error" in self.result:
                return self.result["error"], None  # Return the error and no image
            return "Image generated successfully!", self.result

        except Exception as e:
            return str(e), None

    def generate_image_api(self, prompt, seed, num_steps, height, width, max_seq_length):
        try:
            # Validate height and width
            self.validate_dimensions(height, width)

            image = self.generate_image(prompt, seed, num_steps, height, width, max_seq_length)
            if image and isinstance(image[1], Image.Image):
                buffered = BytesIO()
                image[1].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return {"image": img_str}
            elif isinstance(image, tuple) and image[0]:
                return {"error": image[0]}  # Return the error directly
            else:
                return {"error": "Image generation failed or task queue is busy."}

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    flux_schnell = Flux_Schnell()

    threading.Thread(target=flux_schnell.process_queue, daemon=True).start()

    # UI Interface
    ui_interface = gr.Interface(
        fn=flux_schnell.generate_image,  # Function for UI usage
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
            gr.Number(value=42, label="Seed"),
            gr.Slider(minimum=1, maximum=50, value=4, step=1, label="Number of Inference Steps"),
            gr.Slider(minimum=64, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=8, label="Height"),
            gr.Slider(minimum=64, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=8, label="Width"),
            gr.Slider(minimum=1, maximum=512, value=DEFAULT_MAX_SEQ_LENGTH, step=1, label="Max Sequence Length"),
        ],
        outputs=[gr.Textbox(label="Status"), gr.Image(type="pil")],  # Return status and image
        title="FLUX Schnell Image Generation",
        description="Enter the prompt and parameters to generate an image:"
    )

    # API Interface
    api_interface = gr.Interface(
        fn=flux_schnell.generate_image_api,  # Function for API usage
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
            gr.Number(value=42, label="Seed"),
            gr.Slider(minimum=1, maximum=50, value=4, step=1, label="Number of Inference Steps"),
            gr.Slider(minimum=64, maximum=1024, value=DEFAULT_IMAGE_HEIGHT, step=8, label="Height"),
            gr.Slider(minimum=64, maximum=1024, value=DEFAULT_IMAGE_WIDTH, step=8, label="Width"),
            gr.Slider(minimum=1, maximum=512, value=DEFAULT_MAX_SEQ_LENGTH, step=1, label="Max Sequence Length"),
        ],
        outputs="json",  # Return JSON output for API, including error messages
    )

    # Combine UI and API into a single app with tabs
    gr.TabbedInterface([ui_interface, api_interface], ["UI", "API"]).launch(
        server_name="0.0.0.0", 
        server_port=default_port, 
        share=False
    )
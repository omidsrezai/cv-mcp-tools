import base64
import os
import queue
import re
import threading
from io import BytesIO

import gradio as gr
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Separate ports for UI and API
default_port_ui = int(os.getenv("PORT_UI", 6080))
default_port_api = int(os.getenv("PORT_API", 6081))
default_model_type = os.getenv("MODEL_TYPE", "Qwen-VL")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace newlines and extra spaces with a single space
    text = re.sub(r'<\|.*?\|>', '', text)  # Remove special markers like <|im_end|>
    return text.strip()

class OCRService:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()
        self.result = None
        self.current_model = None
        self.qwen_model = None
        self.qwen_processor = None
        self.janus_model = None
        self.janus_processor = None
        self.janus_tokenizer = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Preload Janus at startup
        self.load_janus_model()
        self.current_model = "Janus"  # Start with Janus preloaded

    def load_qwen_model(self):
        """Unload Janus (if loaded), then load Qwen-VL"""
        if self.current_model == "Qwen-VL":
            return  # Already loaded
        
        self.unload_current_model()
        print("🔄 Loading Qwen-VL model...")

        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.qwen_processor  = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
        
        self.current_model = "Qwen-VL"
        print("✅ Qwen-VL model loaded.")

    def load_janus_model(self):
        """Unload Qwen-VL (if loaded), then load Janus"""
        if self.current_model == "Janus":
            return  # Already loaded
        
        self.unload_current_model()
        print("🔄 Loading Janus model...")

        self.janus_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-7B")
        self.janus_tokenizer = self.janus_processor.tokenizer

        self.janus_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/Janus-Pro-7B", trust_remote_code=True
        )
        self.janus_model = self.janus_model.to(torch.bfloat16).cuda().eval()

        self.current_model = "Janus"
        print("✅ Janus model loaded.")

    def unload_current_model(self):
        """Safely unload the currently loaded model to free memory"""
        print(f"🛑 Unloading {self.current_model} model...")
        if self.current_model == "Qwen-VL":
            del self.qwen_model
            del self.qwen_processor
            self.qwen_model = None
            self.qwen_processor = None
        elif self.current_model == "Janus":
            del self.janus_model
            del self.janus_processor
            del self.janus_tokenizer
            self.janus_model = None
            self.janus_processor = None
            self.janus_tokenizer = None
        
        torch.cuda.empty_cache()  # Free GPU memory
        print("✅ Model unloaded.")

    def run_inference_QwenVL(self, image, prompt, max_tokens=256):
        self.load_qwen_model()  # Ensure Qwen-VL is loaded
        with self.lock:
            try:
                text_input = self.qwen_processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image", "image": "my_image.png"}, 
                                                  {"type": "text", "text": prompt}]}],
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.qwen_processor(
                    text=[text_input],
                    images=[image],
                    videos=None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=max_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return clean_text(output_text[0]) if output_text else "Failed to generate response."

            except Exception as e:
                return f"Error: {str(e)}"

    def run_inference_Janus(self, image, prompt, max_tokens=256):
        self.load_janus_model() 
        with self.lock:
            try:
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt}",
                        "images": ["2.png"],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]

                prepare_inputs = self.janus_processor(
                    conversations=conversation, images=[image], force_batchify=True
                ).to(self.janus_model.device)

                # # run image encoder to get the image embeddings
                inputs_embeds = self.janus_model.prepare_inputs_embeds(**prepare_inputs)

                # # run the model to get the response
                outputs = self.janus_model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.janus_tokenizer.eos_token_id,
                    bos_token_id=self.janus_tokenizer.bos_token_id,
                    eos_token_id=self.janus_tokenizer.eos_token_id,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                return self.janus_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            except Exception as e:
                return f"Error: {str(e)}"

    def process_queue(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            image, prompt, max_tokens, model_type, callback = task
            try:
                if model_type == "Qwen-VL":
                    result = self.run_inference_QwenVL(image, prompt, max_tokens)
                elif model_type == "Janus":
                    result = self.run_inference_Janus(image, prompt, max_tokens)
                else:
                    result = "Invalid model type selected."

                callback(result)
            except Exception as e:
                callback({"error": str(e)})
            finally:
                self.task_queue.task_done()

    def generate_text_ui(self, image, prompt, max_tokens=256, model_type=default_model_type):
        """For UI: accepts a PIL image and model selection."""
        try:
            if not self.task_queue.empty():
                return "Task queue is busy, please try again later.", None

            def process_result(result):
                self.result = result

            self.task_queue.put((image, prompt, max_tokens, model_type, process_result))
            self.task_queue.join()

            if isinstance(self.result, dict) and "error" in self.result:
                return self.result["error"], None
            return "Generated successfully!", self.result

        except Exception as e:
            return str(e), None

    def generate_text_api(self, json_input):
        """For API: accepts a JSON input with base64 image and model selection."""
        try:
            image_base64 = json_input.get("image")
            prompt = json_input.get("prompt", "Describe this image")
            max_tokens = json_input.get("max_tokens", 256)
            model_type = json_input.get("model_type", default_model_type) 

            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data)).convert("RGB")

            result = self.generate_text_ui(image, prompt, max_tokens, model_type)
            if result and isinstance(result[1], str):
                return {"text": result[1]}
            elif isinstance(result, tuple) and result[0]:
                return {"error": result[0]}
            else:
                return {"error": "Failed to generate response."}

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    ocr_service = OCRService()

    threading.Thread(target=ocr_service.process_queue, daemon=True).start()

    # UI Interface (Accepts PIL Image)
    ui_interface = gr.Interface(
        fn=ocr_service.generate_text_ui,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),  # Accepts PIL images
            gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
            gr.Slider(minimum=1, maximum=512, value=256, step=1, label="Max Tokens"),
            gr.Radio(["Janus", "Qwen-VL"], label="Select Model", value=default_model_type),  # Model selection
        ],
        outputs=[gr.Textbox(label="Status"), gr.Textbox(label="Generated Text")],
        title="OCR Server",
        description="Upload an image and enter your prompt."
    )

    # API Interface (Accepts JSON with Base64 Image)
    api_interface = gr.Interface(
        fn=ocr_service.generate_text_api,
        inputs=[gr.JSON(label="JSON Input")],  # Accepts raw JSON input
        outputs=gr.JSON(label="API Response"),  # Returns raw JSON output
        title="OCR API",
        description="API interface for generating an image description."
    )

    # Function to launch UI Interface
    def launch_ui():
        ui_interface.launch(
            server_name="0.0.0.0",
            server_port=default_port_ui,  # Port for the UI
            share=False
        )

    # Function to launch API Interface
    def launch_api():
        api_interface.launch(
            server_name="0.0.0.0",
            server_port=default_port_api,  # Port for the API
            share=False
        )

    # Run UI and API interfaces in separate threads
    threading.Thread(target=launch_ui).start()
    threading.Thread(target=launch_api).start()
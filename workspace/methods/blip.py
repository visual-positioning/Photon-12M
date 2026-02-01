import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List

class CaptionModel:
    def __init__(self, device='cuda', checkpoint_path=None):
        """
        Initializes the BLIP model.
        
        Args:
            device (str): Device to run inference on ('cuda' or 'cpu').
            checkpoint_path (str): Optional. Kept for compatibility with the generic interface,
                                   though standard BLIP loads from HuggingFace.
        """
        # Determine actual device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = "Salesforce/blip-image-captioning-base"
        
        print(f"   [BLIP] Loading model ({self.model_name}) to {self.device}...")
        
        # Load Processor and Model from HuggingFace Hub
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"   [Error] Failed to load BLIP weights: {e}")
            raise e

    def generate_batch(self, image_list: List[Image.Image]) -> List[str]:
        """
        High-performance batch generation.
        
        Args:
            image_list: A list of PIL Images.
            
        Returns:
            List of caption strings corresponding to the inputs.
        """
        # 1. Feature Extraction (Processor handles list of images automatically)
        inputs = self.processor(images=image_list, return_tensors="pt").to(self.device)

        # 2. Generation
        # Using beam search (num_beams=5) is standard for research evaluation 
        # to ensure higher quality metrics than greedy search.
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                num_beams=5, 
                repetition_penalty=1.0,
                length_penalty=1.0
            )

        # 3. Decoding
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Optional: Basic post-processing (strip whitespace)
        captions = [cap.strip() for cap in captions]
        
        return captions

    def generate(self, image: Image.Image) -> str:
        """
        Fallback for single image inference.
        """
        # Reuse batch logic for consistency
        return self.generate_batch([image])[0]
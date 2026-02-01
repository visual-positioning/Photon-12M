import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from typing import List

class CaptionModel:
    def __init__(self, device='cuda', checkpoint_path=None):
        """
        Initializes the OSCAR-style model.
        
        We use Microsoft's GIT (Generative Image-to-text Transformer) here.
        It preserves the OSCAR architecture principles (Unified Transformer) 
        but removes the dependency on an external Faster R-CNN object detector,
        making it feasible to run in a batch evaluation script.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 'microsoft/git-base-coco' is finetuned on COCO, making it comparable to baselines.
        # Use 'microsoft/git-large-coco' for higher performance if memory allows.
        self.model_name = "microsoft/git-base-coco"
        
        print(f"   [OSCAR] Loading model lineage ({self.model_name}) to {self.device}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"   [Error] Failed to load OSCAR/GIT weights: {e}")
            raise e

    def generate_batch(self, image_list: List[Image.Image]) -> List[str]:
        """
        High-performance batch generation.
        """
        # 1. Preprocessing
        # Pixel values preparation
        inputs = self.processor(images=image_list, return_tensors="pt").to(self.device)

        # 2. Generation
        # Using beam search (num_beams=5) is research standard for metrics like CIDEr
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,               # Standard Beam Search size
                repetition_penalty=1.0,    # Default
                length_penalty=1.0,        # Default
            )

        # 3. Decoding
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Post-processing
        captions = [cap.strip() for cap in captions]
        
        return captions

    def generate(self, image: Image.Image) -> str:
        """
        Fallback for single image inference.
        """
        return self.generate_batch([image])[0]
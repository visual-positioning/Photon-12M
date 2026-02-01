import os
import json
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class CaptionDataset(Dataset):
    def __init__(self, root, split='val', mode='full'):
        """
        Args:
            root (str): Root directory (e.g., /path/to/COCO2017).
                        Expected structure:
                        root/
                          annotations/
                          train2017/
                          val2017/
                          test2017/
            split (str): 'train', 'val', or 'test'.
            mode (str): 'full' (returns image + captions) or 'captions_only' (returns only captions).
        """
        self.root = root
        self.split = split.lower()
        self.mode = mode
        self.samples = []
        self.has_gt = True # Flag to check if ground truth exists

        # 1. Determine Paths based on split
        # COCO naming convention usually follows 'train2017', 'val2017', 'test2017'
        if 'train' in self.split:
            self.img_dir = os.path.join(root, 'train2017')
            self.ann_file = os.path.join(root, 'annotations', 'captions_train2017.json')
        elif 'val' in self.split:
            self.img_dir = os.path.join(root, 'val2017')
            self.ann_file = os.path.join(root, 'annotations', 'captions_val2017.json')
        elif 'test' in self.split:
            self.img_dir = os.path.join(root, 'test2017')
            # Test sets usually have 'image_info' but NO captions
            self.ann_file = os.path.join(root, 'annotations', 'image_info_test2017.json')
            self.has_gt = False
        else:
            raise ValueError(f"Unknown split: {split}")

        # 2. Load Metadata
        if not os.path.exists(self.ann_file):
            # Fallback for custom naming or missing files
            print(f"[Warning] Annotation file not found: {self.ann_file}")
            # If strictly testing without GT, we might just scan the directory
            self.has_gt = False
            self.samples = self._scan_directory(self.img_dir)
        else:
            self._load_json_annotations()

        print(f"   [COCO] Loaded {self.split.upper()} set. Samples: {len(self.samples)}. GT Available: {self.has_gt}")

    def _load_json_annotations(self):
        """Parses COCO standard JSON format."""
        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        # Map image_id -> filename
        img_map = {img['id']: img['file_name'] for img in data['images']}
        
        # Map image_id -> list of captions (If GT exists)
        caps_map = defaultdict(list)
        if self.has_gt and 'annotations' in data:
            for ann in data['annotations']:
                caps_map[ann['image_id']].append(ann['caption'])

        # Build final registry
        # We iterate over images to ensure we include those even without captions (if any)
        for img_id, filename in img_map.items():
            self.samples.append({
                "image_id": img_id,
                "file_name": filename,
                "captions": caps_map[img_id] if self.has_gt else [] # Handle missing GT
            })

    def _scan_directory(self, folder):
        """Fallback: Just reads image files if no JSON is found (Blind Test)."""
        samples = []
        if not os.path.exists(folder):
            print(f"[Error] Image directory not found: {folder}")
            return []
            
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Use filename as dummy ID
                samples.append({
                    "image_id": fname,
                    "file_name": fname,
                    "captions": []
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        result = {
            "image_id": item['image_id'],
            "captions": item['captions']
        }

        # If we need the image (standard eval/train)
        if self.mode == 'full':
            img_path = os.path.join(self.img_dir, item['file_name'])
            try:
                # Open and convert to RGB (standardize 3 channels)
                image = Image.open(img_path).convert('RGB')
                result["image"] = image
            except Exception as e:
                print(f"[Error] Could not load image {img_path}: {e}")
                # Return None or a dummy black image to prevent crash
                result["image"] = Image.new('RGB', (224, 224))
        
        return result

    def collate_fn(self, batch):
        """
        Custom collate is required because 'captions' is a list (variable length)
        and 'image' is a PIL object (not yet a tensor).
        """
        batch_out = {
            "image_id": [x['image_id'] for x in batch],
            "captions": [x['captions'] for x in batch]
        }
        
        # Only collate images if they were requested
        if 'image' in batch[0]:
            batch_out["image"] = [x['image'] for x in batch]
            
        return batch_out
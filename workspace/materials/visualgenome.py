"""
Visual Genome Dataset Loader for Image Captioning Evaluation.

Visual Genome contains dense annotations including region descriptions.
For captioning evaluation, we aggregate region descriptions as references
or use image-level captions if available.

Expected directory structure:
    VISUALGENOME/
        images/
            VG_100K/
                *.jpg
            VG_100K_2/
                *.jpg
        region_descriptions.json (OR)
        image_data.json + region_descriptions.json
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import random


class CaptionDataset(Dataset):
    def __init__(self, root, split='val', mode='full', max_regions_per_image=5):
        """
        Args:
            root (str): Root directory (e.g., /path/to/VISUALGENOME).
            split (str): 'train', 'val', or 'test' (custom splits since VG doesn't have official ones).
            mode (str): 'full' (image + captions) or 'captions_only'.
            max_regions_per_image (int): Max number of region descriptions to use as captions.
        """
        self.root = root
        self.split = split.lower()
        self.mode = mode
        self.max_regions = max_regions_per_image
        self.samples = []
        self.has_gt = True
        
        # Determine image directories (VG has two image folders)
        self.img_dirs = [
            os.path.join(root, 'images', 'VG_100K'),
            os.path.join(root, 'images', 'VG_100K_2'),
            os.path.join(root, 'images'),  # Fallback
            root  # Final fallback
        ]
        self.img_dirs = [d for d in self.img_dirs if os.path.exists(d)]
        
        # Load annotations
        region_file = os.path.join(root, 'region_descriptions.json')
        
        if os.path.exists(region_file):
            self._load_region_descriptions(region_file)
        else:
            print(f"[Warning] region_descriptions.json not found in {root}")
            self.has_gt = False
            self.samples = self._scan_all_image_dirs()
        
        # Apply split (VG doesn't have official splits, we create them)
        self._apply_split()
        
        print(f"   [VisualGenome] Loaded {self.split.upper()} set. Samples: {len(self.samples)}. GT Available: {self.has_gt}")
    
    def _load_region_descriptions(self, filepath):
        """Load Visual Genome region descriptions and aggregate per image."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Aggregate regions per image
        img_caps = defaultdict(list)
        
        for img_data in data:
            img_id = img_data['id']
            
            # Extract region phrases
            for region in img_data.get('regions', []):
                phrase = region.get('phrase', '').strip()
                if phrase and len(phrase) > 10:  # Filter very short phrases
                    img_caps[img_id].append(phrase)
        
        # Build samples
        for img_id, captions in img_caps.items():
            # Use top N longest/most descriptive regions
            captions = sorted(captions, key=len, reverse=True)[:self.max_regions]
            
            if captions:  # Only include if we have at least one caption
                self.samples.append({
                    'image_id': img_id,
                    'file_name': f"{img_id}.jpg",
                    'captions': captions
                })
    
    def _apply_split(self):
        """Apply train/val/test split to Visual Genome (80/10/10 by default)."""
        if not self.samples:
            return
        
        # Deterministic shuffle based on image_id
        self.samples.sort(key=lambda x: x['image_id'])
        
        n = len(self.samples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        if self.split == 'train':
            self.samples = self.samples[:train_end]
        elif self.split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif self.split == 'test':
            self.samples = self.samples[val_end:]
        # 'all' or unknown keeps all samples
    
    def _scan_all_image_dirs(self):
        """Scan all image directories for images."""
        samples = []
        seen = set()
        
        for img_dir in self.img_dirs:
            if not os.path.exists(img_dir):
                continue
            
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_id = fname.split('.')[0]
                    if img_id not in seen:
                        seen.add(img_id)
                        samples.append({
                            'image_id': img_id,
                            'file_name': fname,
                            'captions': []
                        })
        
        return samples
    
    def _find_image_path(self, filename):
        """Find the actual path to an image across multiple directories."""
        for img_dir in self.img_dirs:
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                return path
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        result = {
            'image_id': item['image_id'],
            'captions': item['captions']
        }
        
        if self.mode == 'full':
            img_path = self._find_image_path(item['file_name'])
            try:
                if img_path:
                    image = Image.open(img_path).convert('RGB')
                else:
                    raise FileNotFoundError(f"Image not found: {item['file_name']}")
                result['image'] = image
            except Exception as e:
                print(f"[Error] Could not load image {item['file_name']}: {e}")
                result['image'] = Image.new('RGB', (224, 224))
        
        return result
    
    def collate_fn(self, batch):
        """Custom collate for variable-length captions and PIL images."""
        batch_out = {
            'image_id': [x['image_id'] for x in batch],
            'captions': [x['captions'] for x in batch]
        }
        
        if 'image' in batch[0]:
            batch_out['image'] = [x['image'] for x in batch]
        
        return batch_out

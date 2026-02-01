"""
NoCaps Dataset Loader for Image Captioning Evaluation.

NoCaps (Novel Object Captioning at Scale) is designed to test generalization
to novel object classes not seen during training.

Expected directory structure:
    NOCAPS/
        images/
            val/
                *.jpg
        nocaps_val_4500_captions.json
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class CaptionDataset(Dataset):
    def __init__(self, root, split='val', mode='full', domain=None):
        """
        Args:
            root (str): Root directory (e.g., /path/to/NOCAPS).
            split (str): 'val' or 'test' (test has no public GT).
            mode (str): 'full' (image + captions) or 'captions_only'.
            domain (str): Optional filter: 'in-domain', 'near-domain', 'out-of-domain', or None for all.
        """
        self.root = root
        self.split = split.lower()
        self.mode = mode
        self.domain = domain
        self.samples = []
        self.has_gt = True
        
        # Determine paths
        self.img_dir = os.path.join(root, 'images', self.split)
        if not os.path.exists(self.img_dir):
            # Fallback: images might be directly in root/images
            self.img_dir = os.path.join(root, 'images')
        
        # Annotation file
        ann_file = os.path.join(root, f'nocaps_{self.split}_4500_captions.json')
        if not os.path.exists(ann_file):
            # Try alternative naming
            ann_file = os.path.join(root, 'annotations', f'nocaps_{self.split}.json')
        
        if os.path.exists(ann_file):
            self._load_nocaps_json(ann_file)
        else:
            print(f"[Warning] NoCaps annotation file not found: {ann_file}")
            self.has_gt = False
            self.samples = self._scan_directory(self.img_dir)
        
        print(f"   [NoCaps] Loaded {self.split.upper()} set. Samples: {len(self.samples)}. GT Available: {self.has_gt}")
    
    def _load_nocaps_json(self, filepath):
        """Load NoCaps JSON format (COCO-style with domain info)."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Build image info map
        img_info = {}
        for img in data['images']:
            img_info[img['id']] = {
                'file_name': img['file_name'],
                'domain': img.get('domain', 'unknown')  # in-domain, near-domain, out-of-domain
            }
        
        # Build caption map
        caps_map = defaultdict(list)
        if 'annotations' in data:
            for ann in data['annotations']:
                caps_map[ann['image_id']].append(ann['caption'])
        else:
            self.has_gt = False
        
        # Build samples with optional domain filtering
        for img_id, info in img_info.items():
            if self.domain and info['domain'] != self.domain:
                continue
            
            self.samples.append({
                'image_id': img_id,
                'file_name': info['file_name'],
                'captions': caps_map[img_id] if self.has_gt else [],
                'domain': info['domain']
            })
    
    def _scan_directory(self, folder):
        """Fallback: Scan directory for images."""
        samples = []
        if not os.path.exists(folder):
            print(f"[Error] Image directory not found: {folder}")
            return []
        
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'image_id': fname.split('.')[0],
                    'file_name': fname,
                    'captions': [],
                    'domain': 'unknown'
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        result = {
            'image_id': item['image_id'],
            'captions': item['captions'],
            'domain': item.get('domain', 'unknown')
        }
        
        if self.mode == 'full':
            img_path = os.path.join(self.img_dir, item['file_name'])
            try:
                image = Image.open(img_path).convert('RGB')
                result['image'] = image
            except Exception as e:
                print(f"[Error] Could not load image {img_path}: {e}")
                result['image'] = Image.new('RGB', (224, 224))
        
        return result
    
    def collate_fn(self, batch):
        """Custom collate for variable-length captions and PIL images."""
        batch_out = {
            'image_id': [x['image_id'] for x in batch],
            'captions': [x['captions'] for x in batch],
            'domain': [x.get('domain', 'unknown') for x in batch]
        }
        
        if 'image' in batch[0]:
            batch_out['image'] = [x['image'] for x in batch]
        
        return batch_out

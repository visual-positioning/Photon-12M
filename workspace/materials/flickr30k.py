"""
Flickr30K Dataset Loader for Image Captioning Evaluation.

Expected directory structure:
    FLICKR30K/
        images/
            1000092795.jpg
            1000268201.jpg
            ...
        results_20130124.token  (OR)
        dataset_flickr30k.json  (Karpathy split)
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class CaptionDataset(Dataset):
    def __init__(self, root, split='test', mode='full'):
        """
        Args:
            root (str): Root directory (e.g., /path/to/FLICKR30K).
            split (str): 'train', 'val', or 'test' (for Karpathy splits).
            mode (str): 'full' (image + captions) or 'captions_only'.
        """
        self.root = root
        self.split = split.lower()
        self.mode = mode
        self.samples = []
        self.has_gt = True
        
        # Determine image directory
        self.img_dir = os.path.join(root, 'images')
        if not os.path.exists(self.img_dir):
            # Fallback: images might be directly in root
            self.img_dir = root
        
        # Try different annotation formats
        karpathy_file = os.path.join(root, 'dataset_flickr30k.json')
        token_file = os.path.join(root, 'results_20130124.token')
        
        if os.path.exists(karpathy_file):
            self._load_karpathy_split(karpathy_file)
        elif os.path.exists(token_file):
            self._load_token_file(token_file)
        else:
            print(f"[Warning] No annotation file found in {root}")
            self.has_gt = False
            self.samples = self._scan_directory(self.img_dir)
        
        print(f"   [Flickr30K] Loaded {self.split.upper()} set. Samples: {len(self.samples)}. GT Available: {self.has_gt}")
    
    def _load_karpathy_split(self, filepath):
        """Load Karpathy JSON split format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for img_data in data['images']:
            # Filter by split
            if img_data['split'] != self.split:
                continue
            
            img_id = img_data.get('imgid', img_data['filename'].split('.')[0])
            filename = img_data['filename']
            
            # Extract captions
            captions = [sent['raw'] for sent in img_data.get('sentences', [])]
            
            self.samples.append({
                'image_id': img_id,
                'file_name': filename,
                'captions': captions
            })
    
    def _load_token_file(self, filepath):
        """Load original Flickr30K .token file format."""
        caps_map = defaultdict(list)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: 1000092795.jpg#0\tTwo young guys with shaggy hair...
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                img_ref = parts[0]  # 1000092795.jpg#0
                caption = parts[1]
                
                # Extract image filename
                img_name = img_ref.split('#')[0]
                caps_map[img_name].append(caption)
        
        # Build samples (Flickr30K doesn't have official splits in token file)
        # Use all for evaluation unless split filtering needed
        img_list = list(caps_map.keys())
        
        # Simple split logic: test=last 1000, val=prev 1000, train=rest
        if self.split == 'test':
            img_list = img_list[-1000:]
        elif self.split == 'val':
            img_list = img_list[-2000:-1000]
        elif self.split == 'train':
            img_list = img_list[:-2000]
        
        for img_name in img_list:
            img_id = img_name.split('.')[0]
            self.samples.append({
                'image_id': img_id,
                'file_name': img_name,
                'captions': caps_map[img_name]
            })
    
    def _scan_directory(self, folder):
        """Fallback: Just reads image files if no annotations found."""
        samples = []
        if not os.path.exists(folder):
            print(f"[Error] Image directory not found: {folder}")
            return []
        
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'image_id': fname.split('.')[0],
                    'file_name': fname,
                    'captions': []
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        result = {
            'image_id': item['image_id'],
            'captions': item['captions']
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
            'captions': [x['captions'] for x in batch]
        }
        
        if 'image' in batch[0]:
            batch_out['image'] = [x['image'] for x in batch]
        
        return batch_out

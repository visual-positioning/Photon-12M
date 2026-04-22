import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Assuming you are using the official apple/ml-mobileclip repository
# If you are using a custom wrapper, adjust the import accordingly.
import mobileclip

# ==========================================
# 1. CONFIGURATION
# ==========================================
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map your dataset to its images directory and annotation file
# Update the "image_dir" paths to where the actual JPGs are stored.
DATASETS_CONFIG = {
    "COCO_karpathy_test": {
        "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_coco.json"),
        "image_dir": os.path.expanduser("~/dataset/MSCOCO2014/val2014"), # Usually karpathy test uses val2014 images
        "save_path": os.path.expanduser("~/dataset/MSCOCO2014/embeddings_test.pt"),
        "format": "karpathy"
    },
    "Flickr8k": {
        "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_flickr8k.json"),
        "image_dir": os.path.expanduser("~/dataset/flickr8k/Images"), # Verify this points to the JPGs
        "save_path": os.path.expanduser("~/dataset/flickr8k/embeddings_test.pt"),
        "format": "karpathy"
    },
    "Flickr30k": {
        "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_flickr30k.json"),
        "image_dir": os.path.expanduser("~/dataset/flickr30k_images/flickr30k_images"), # Verify this points to the JPGs
        "save_path": os.path.expanduser("~/dataset/flickr30k_images/embeddings_test.pt"),
        "format": "karpathy"
    },
        "TextCaps": {
        "annot_path": os.path.expanduser("~/dataset/TextCaps/TextCaps_0.1_val.json"),
        "image_dir": os.path.expanduser("~/dataset/TextCaps/train_images"), 
        "save_path": os.path.expanduser("~/dataset/TextCaps/embeddings_val.pt"),
        "format": "textcaps" 
    },
    "NoCaps": {
        "annot_path": os.path.expanduser("~/dataset/nocaps/nocaps_val_captions.json"),
        "image_dir": os.path.expanduser("~/dataset/nocaps/val"),
        "save_path": os.path.expanduser("~/dataset/nocaps/embeddings_val.pt"),
        "format": "coco_standard"
    }
}

# ==========================================
# 2. DATASET PARSING & LOADING
# ==========================================
def get_image_paths_and_ids(config):
    """
    Parses the annotation JSON to extract required image IDs and their corresponding filenames.
    """
    image_list = []
    with open(config["annot_path"], 'r') as f:
        data = json.load(f)
        
    if config["format"] == "karpathy":
        for img in data['images']:
            if img['split'] == 'test':
                # COCO uses 'cocoid', Flickr uses 'imgid'
                image_id = img.get('cocoid', img.get('imgid'))
                
                # Ultimate fallback just in case some custom split drops both keys
                if image_id is None:
                    image_id = img['filename'] 
                    
                image_list.append({
                    "id": image_id, 
                    "filename": img['filename']
                })
                
    elif config["format"] == "coco_standard":
        for img in data['images']:
            image_list.append({
                "id": img['id'],
                "filename": img['file_name']
            })
            
    elif config["format"] == "textcaps":
        # TextCaps stores image info under the 'data' key
        for img in data['data']:
            image_list.append({
                "id": img['image_id'],
                "filename": img['image_name'] + ".jpg" # TextCaps often omits the extension
            })
            
    return image_list
class ImageExtractionDataset(Dataset):
    def __init__(self, image_list, image_dir, transform):
        self.image_list = image_list
        self.image_dir = image_dir
        self.transform = transform
        
        # Filter out missing images to prevent crashes
        self.valid_images = []
        for img_data in self.image_list:
            full_path = os.path.join(self.image_dir, img_data["filename"])
            if os.path.exists(full_path):
                img_data["full_path"] = full_path
                self.valid_images.append(img_data)
            else:
                print(f"Warning: Image missing and will be skipped -> {full_path}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_data = self.valid_images[idx]
        image_id = img_data["id"]
        
        try:
            # Load and convert to RGB
            image = Image.open(img_data["full_path"]).convert('RGB')
            tensor_image = self.transform(image)
            return image_id, tensor_image
        except Exception as e:
            print(f"Error loading {img_data['full_path']}: {e}")
            # Return a dummy tensor if it fails, though filtering above usually prevents this
            return image_id, torch.zeros((3, 256, 256)) 

# ==========================================
# 3. EXTRACTION ROUTINE
# ==========================================
def extract_and_save_embeddings(dataset_name, config, model, transform):
    print(f"\n--- Processing {dataset_name} ---")
    
    # 1. Map IDs to files
    image_list = get_image_paths_and_ids(config)
    print(f"Found {len(image_list)} images defined in {dataset_name} annotations.")
    
    # 2. Create Dataset & Loader
    dataset = ImageExtractionDataset(image_list, config["image_dir"], transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    
    print(f"Successfully located {len(dataset)} valid image files on disk.")
    
    embeddings_dict = {}
    
    # 3. Extract Embeddings
    with torch.no_grad():
        for image_ids, images in tqdm(loader, desc="Extracting"):
            images = images.to(DEVICE)
            
            # Get MobileCLIP image features
            # Note: Depending on your exact mobileclip version, you might need to adjust this call.
            # Usually it returns un-normalized features, or you explicitly call encode_image
            image_features = model.encode_image(images)
            
            # Normalize features if required by your decoder architecture
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Move to CPU immediately to save RAM
            image_features = image_features.cpu()
            
            # Store in dictionary using the image_id
            for i, img_id in enumerate(image_ids):
                # Ensure img_id is a native python type (int or str), not a tensor
                if isinstance(img_id, torch.Tensor):
                    img_id = img_id.item() 
                embeddings_dict[img_id] = image_features[i]
                
    # 4. Save to disk
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    torch.save(embeddings_dict, config["save_path"])
    print(f"✅ Saved embeddings dictionary of size {len(embeddings_dict)} to {config['save_path']}")


def main():
    print(f"Loading MobileCLIP-S1 to {DEVICE}...")
    # Load the model. Change the weights path/identifier to match your local setup
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='models/mobileclip_s1.pt')
    model = model.to(DEVICE)
    model.eval()

    for dataset_name, config in DATASETS_CONFIG.items():
        if os.path.exists(config["save_path"]):
            print(f"\n⏭️ Skipping {dataset_name} - Embeddings already exist at {config['save_path']}")
            continue
            
        try:
            extract_and_save_embeddings(dataset_name, config, model, preprocess)
        except Exception as e:
            print(f"❌ Failed processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    # Optimize for Ampere/Ada architectures if you are using an RTX 4060 / 2080Ti
    torch.backends.cudnn.benchmark = True
    main()

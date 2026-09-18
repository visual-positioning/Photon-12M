# Precomputed Image Embeddings

To accelerate training, Photon precomputes MobileCLIP-S1 image embeddings.

## Extracting Embeddings

Run the extraction script to process images and save their L2-normalized MobileCLIP embeddings:

```bash
python datapreprocessing/extract_coco2014_karapathy_embedding.py
```

* **Input format:** MS-COCO images.
* **Output format:** PyTorch `.pt` files containing a dictionary mapping filenames to FP16 embeddings.
* **Storage location:** Saved in the `embeddings/` subdirectory of your COCO dataset path.

The training script `training/train_nano.py` loads these embeddings directly from the specified directory.

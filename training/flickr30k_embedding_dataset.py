import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class Flickr30kEmbeddingDataset(Dataset):
    """
    Flickr30k dataset using precomputed MobileCLIP embeddings.
    - Split by IMAGE first (Karpathy split)
    - Then expand image -> multiple (embedding, caption) pairs
    """

    def __init__(
        self,
        embedding_root: str,
        karpathy_json: str,
        split: str,
        tokenizer,
        max_length: int = 32,
    ):
        """
        Args:
            embedding_root: path to split folder
                e.g. datapreprocessing/flickr30k_embeddings_s1/train
            karpathy_json: dataset_flickr30k.json
            split: 'train' | 'val' | 'test'
            tokenizer: existing tokenizer (COCO tokenizer)
            max_length: max caption length
        """
        assert split in ["train", "val", "test"]

        self.embedding_root = embedding_root
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        # --------------------------------------------------
        # 1. Load Karpathy JSON and collect captions per image
        # --------------------------------------------------
        with open(karpathy_json, "r") as f:
            data = json.load(f)

        # filename -> list[captions]
        image_to_captions = {}

        for item in data["images"]:
            if item["split"] != split:
                continue

            captions = [s["raw"].strip() for s in item["sentences"]]
            image_to_captions[item["filename"]] = captions

        # --------------------------------------------------
        # 2. Load embedding chunks and build samples
        # --------------------------------------------------
        self.samples = []
        self._load_embeddings(image_to_captions)

        print(
            f"[Flickr30k] split={split} | samples={len(self.samples)}"
        )

    def _load_embeddings(self, image_to_captions):
        """
        Builds (embedding, caption) pairs AFTER split
        """
        chunk_files = sorted(
            f for f in os.listdir(self.embedding_root)
            if f.endswith(".npz")
        )

        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.embedding_root, chunk_file)
            data = np.load(chunk_path)

            embeddings = data["embeddings"]  # (N, 512)
            paths = data["paths"]            # (N,)

            for emb, fname in zip(embeddings, paths):
                if fname not in image_to_captions:
                    continue  # safety check

                for caption in image_to_captions[fname]:
                    self.samples.append((emb, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embedding, caption = self.samples[idx]

        # Convert embedding to tensor
        embedding = torch.tensor(embedding, dtype=torch.float32)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)

        return embedding, input_ids

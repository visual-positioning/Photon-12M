"""
coco2014_tokenizer.py

Train a BPE tokenizer ONLY on MSCOCO 2014 Karpathy TRAIN captions.
This avoids linguistic data leakage and is mandatory for paper submission.
"""

import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# ==========================================
# CONFIGURATION (YOUR PATHS)
# ==========================================
KARPATHY_JSON = "/scratch/kalidas_5/Dataset/karapathy_split/dataset_coco.json"
VOCAB_SIZE = 8000
SAVE_FILE = "coco2014_tokenizer.json"


# ==========================================
# LOAD KARPATHY TRAIN CAPTIONS ONLY
# ==========================================
def load_train_captions():
    assert os.path.exists(KARPATHY_JSON), "Karpathy split JSON not found!"

    with open(KARPATHY_JSON, "r") as f:
        data = json.load(f)

    captions = []
    for img in data["images"]:
        if img["split"] == "train":
            for sent in img["sentences"]:
                captions.append(sent["raw"])

    assert len(captions) > 0, "No training captions found!"
    print(f"[INFO] Loaded {len(captions)} Karpathy TRAIN captions")

    return captions


# ==========================================
# TRAIN TOKENIZER
# ==========================================
def train_coco_tokenizer():
    captions = load_train_captions()

    # 1. Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Byte-level pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. Byte-level decoding
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True)

    # 4. Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 5. Train tokenizer (THIS IS "TRAINING")
    tokenizer.train_from_iterator(captions, trainer=trainer)

    # 6. Post-processing ([CLS] ... [SEP])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # 7. Save tokenizer
    # os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    tokenizer.save(SAVE_FILE)

    print(
        f"[SUCCESS] Tokenizer saved to {SAVE_FILE} "
        f"(vocab size = {tokenizer.get_vocab_size()})"
    )


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train_coco_tokenizer()

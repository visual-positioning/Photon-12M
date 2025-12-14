import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# CONFIG
# ---------------------
CAPTION_FILE = r"C:\Dataset\coco2017\annotations\captions_train2017.json" # Adjust path
VOCAB_SIZE = 8000  # <--- Key for keeping model small
SAVE_FILE = "tokenizer/mobilecap_tokenizer.json"

def train_coco_tokenizer():
    print(f"Loading captions from {CAPTION_FILE}...")
    with open(CAPTION_FILE, 'r') as f:
        data = json.load(f)

    # Extract just the text strings
    # format: data['annotations'] -> list of dicts with 'caption' key
    captions = [item['caption'] for item in data['annotations']]
    print(f"Found {len(captions)} captions. Training BPE...")

    # 1. Initialize empty BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Pre-processing: Split on whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 3. Decoding: Revert ByteLevel
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Trainer: Define special tokens and vocab size
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. Train
    tokenizer.train_from_iterator(captions, trainer=trainer)

    # 6. Post-Processing (Optional but good for BERT/RoBERTa style)
    # Adds [CLS] at start and [SEP] at end
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # 7. Save
    tokenizer.save(SAVE_FILE)
    print(f"Tokenizer saved to {SAVE_FILE} with vocab size {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_coco_tokenizer()
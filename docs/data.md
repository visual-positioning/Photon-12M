# Dataset Preparation

Photon uses the MS-COCO 2014 dataset with the Karpathy split.

## Dataset Structure

The preprocessing scripts expect the dataset to be organized as follows (configurable in `datapreprocessing/extract_coco2014_karapathy_embedding.py`):

```text
MSCOCO2014/
├── train2014/
├── val2014/
└── embeddings/       (Created by extraction scripts)
```
You also need the Karpathy split annotations JSON: `dataset_coco.json`

## Preprocessing

1. **Images:** Resized so shorter side is 224, followed by a 224x224 center crop, RGB conversion, and normalization.
2. **Captions:** Tokenized using a Byte-Pair Encoding (BPE) tokenizer trained exclusively on the MS-COCO Karpathy training split (`tokenizer/mobilecap_tokenizer.json`). The vocabulary size is 8,000, with a maximum sequence length of 32 tokens. Special tokens: `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.

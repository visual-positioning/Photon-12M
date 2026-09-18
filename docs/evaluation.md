# Evaluation

Photon is evaluated on standard n-gram metrics and semantic alignment metrics.

## Evaluation Command

Run the evaluation script to calculate scores on the MS-COCO Karpathy test split:

```bash
python evaluation/evaluate_tokenizers.py
```
*(Note: Ensure paths in the script are updated to match your local setup).*

## Metrics Evaluated

* **N-gram / Sequence:** BLEU-1 to BLEU-4, METEOR, ROUGE-L, CIDEr
* **Semantic Alignment (via additional tools):** BERTScore, CLIPScore, SBERT

Expected MS-COCO Karpathy results (from paper):
- **CIDEr:** 108.59
- **BLEU-4:** 32.33
- **METEOR:** 26.87
- **ROUGE-L:** 55.06
- **BERTScore-F1:** 65.63

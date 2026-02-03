# Image Captioning Datasets

## MS COCO (2014)
- **Input/Output**: Image → 5 captions per image
- **Download URL**: [MS COCO Dataset](http://mscoco.org/dataset/#download)
- **Citation**: Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., … & Dollár, P. (2014). *Microsoft COCO: Common Objects in Context*. In *ECCV* (pp. 740–755).

## Flickr30K (2014)
- **Input/Output**: Image → 5 captions per image
- **Download URL**: [Kaggle - Flickr30K](https://www.kaggle.com/datasets/banuprasadb/flickr30k)
- **Citation**: Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). *From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions*. *TACL*, 2, 67–78.

## Flickr8K (2013)
- **Input/Output**: Image → 5 captions per image
- **Download URL**: [Kaggle - Flickr8K](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Citation**: Hodosh, M., Young, P., & Hockenmaier, J. (2013). *Framing image description as a ranking task: Data, models and evaluation metrics*. *JAIR*, 47, 853–899.

## Conceptual Captions (2018)
- **Input/Output**: Image → 1 caption per image
- **Download URL**: [Google AI Conceptual Captions](https://ai.google.com/research/ConceptualCaptions)
- **Citation**: Sharma, P., Ding, N., Goodman, S., & Soricut, R. (2018). *Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset for Automatic Image Captioning*. In *Proc. ACL* (pp. 2556–2565).

## Visual Genome (2017)
- **Input/Output**: Image → ~50 region-level captions per image
- **Download URL**: [Visual Genome](https://visualgenome.org/)
- **Citation**: Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., … Fei-Fei, L. (2017). *Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations*. *IJCV*, 123, 32–73.

## SBU Captions (2011)
- **Input/Output**: Image → 1 caption per image
- **Download URL**: [Kaggle - SBU Captions](https://www.kaggle.com/datasets/akashnuka/sbucaptions)
- **Citation**: Ordonez, V., Kulkarni, G., & Berg, T. L. (2011). *Im2Text: Describing Images Using 1 Million Captioned Photographs*. In *NeurIPS 2011*.

---

## Dataset Summary

| Dataset | # Images | Size | # Captions | Train | Test | Val | Split | Remark |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MS COCO 2014** | 164,062 | 27GB | 5 | 82,783 | 5,000 | 5,000 | Karpathy | - |
| **MS COCO 2017** | 163,957 | 27GB | 5 | 118,287 | 40,670 | 5,000 | - | MS COCO Evaluation Server |
| **Flickr30k** | 31,783 | 4.3GB | 5 | 29,000 | 1,000 | 1,000 | Karpathy | - |
| **Flickr8k** | 8,091 | 1.1GB | 5 | 6,000 | 1,000 | 1,000 | Karpathy | - |
| **NoCaps** | 10,600 | 3.2GB | 10 | - | 10,600 | - | - | Zero-shot; train on COCO only |
| **VizWiz** | 39,704 | 18GB | 5 | 23,954 | 8,000 | 7,750 | Official | Test on EvalAI |
| **Visual Genome**| 108,249 | 21GB | ~10-250+ | - | - | - | Random | Region Descriptions |
| **SBU** | 840,407 | 46GB | 1 | 830,407 | 5,000 | 5,000 | Random | Noisy captions |
| **TextCaps** | 28,472 | 7.8GB | 5 | 21,953 | 3,353 | 3,166 | Official | Test on EvalAI |
| **CC3M** | 1,206,390| 36GB | 1 | - | - | - | - | - |

**Notes**
- Karpathy split is commonly used for fair comparison in captioning literature.
- MS COCO 2017 test captions are hidden and evaluated via the official server.
- SBU and CC3M are primarily used for large-scale weakly supervised pretraining.
- Visual Genome provides region-level descriptions, not global image captions.


# Evaluation Metrics for Image Captioning

Evaluation metrics assess generated captions against human references, balancing **n-gram overlap**, **semantic similarity**, and **correlation with human judgments**. Automatic metrics are widely used due to scalability, but **human evaluation remains the gold standard**. A key challenge is that some metrics penalize valid synonyms or novel phrasing.

---

## Automatic Metrics

| Metric Name                                                              | Description                                                                                                    | Strengths / Limitations                                                                                                  | Key Paper / Link                          |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| **BLEU (Bilingual Evaluation Understudy)**                               | N-gram precision (B1–B4: 1–4 grams) with brevity penalty; geometric mean of scores.                            | Simple and standard; correlates with MT but penalizes synonyms and novel phrasing harshly. Often reported as **BLEU-4**. | Papineni et al. (2002): *BLEU*            |
| **METEOR (Metric for Evaluation of Translation with Explicit ORdering)** | Harmonic mean of unigram precision and recall; includes synonym matching, stemming, and paraphrasing.          | Better semantic handling than BLEU; still relies on word overlap.                                                        | Banerjee & Lavie (2005): *METEOR*         |
| **ROUGE-L**                                                              | Longest Common Subsequence (LCS) overlap between candidate and reference captions.                             | Captures sentence-level structure; less granular than n-gram-based metrics.                                              | Lin (2004): *ROUGE*                       |
| **CIDEr (Consensus-based Image Description Evaluation)**                 | TF-IDF weighted n-gram consensus across multiple references with length penalty.                               | Strong correlation with human judgment on COCO; favors informative captions.                                             | Vedantam et al. (2015): *CIDEr*           |
| **SPICE (Semantic Propositional Image Caption Evaluation)**              | Graph-based semantic matching of objects, attributes, and relations using scene graphs; F-score over tuples.   | Strong semantic focus; robust to synonyms; ignores word order and is computationally heavy.                              | Anderson et al. (2016): *SPICE*           |
| **Learned Metrics (e.g., Learned Critique)**                             | Discriminative models trained to distinguish human vs. machine captions; robust to linguistic transformations. | High correlation with human judgments; mitigates overlap bias; requires training data.                                   | Cui et al. (2018): *Learning to Evaluate* |

---

## Human Evaluation Metrics

Human evaluation remains the most reliable way to assess caption quality. Common protocols include:

* **Turing Test pass rate**
* **Percentage of captions rated better or equal to human captions**
* **Expert ratings** (e.g., 1–4 scale used in Flickr8k-Expert)

---

## Tools

* **pycocoevalcap**: Standard evaluation toolkit used for COCO captioning benchmarks.

---

> **Note**: Best practice is to report **multiple automatic metrics (e.g., BLEU-4, METEOR, CIDEr, SPICE)** alongside **human evaluation** when possible.

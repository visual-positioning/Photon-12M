"""
ROUGE-L Metric for Image Captioning Evaluation.

ROUGE-L (Longest Common Subsequence) measures sentence-level structure similarity
by finding the longest co-occurring in-sequence n-grams.

Range: [0, 1] (multiplied by 100 for percentage)
Interpretation: Higher is better. Focuses on recall and structural coherence.
"""
from pycocoevalcap.rouge.rouge import Rouge


class Metric:
    def __init__(self):
        """Initialize ROUGE scorer."""
        self.scorer = Rouge()
    
    def compute(self, candidates, references):
        """
        Compute ROUGE-L score.
        
        Args:
            candidates: {image_id: ["predicted caption"]}
            references: {image_id: ["ref1", "ref2", ...]}
            
        Returns:
            Dictionary with ROUGE-L score (scaled to 0-100).
        """
        score, _ = self.scorer.compute_score(references, candidates)
        
        return {"ROUGE-L": score * 100}

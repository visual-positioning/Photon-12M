"""
METEOR Metric for Image Captioning Evaluation.

METEOR (Metric for Evaluation of Translation with Explicit ORdering) computes
the harmonic mean of unigram precision and recall, with additional features:
- Stemming (run = running)
- Synonymy matching (auto = car)
- Paraphrase matching

Range: [0, 1] (multiplied by 100 for percentage)
Interpretation: Higher is better. Better correlation with human judgment than BLEU
               due to synonym handling.
"""
from pycocoevalcap.meteor.meteor import Meteor


class Metric:
    def __init__(self):
        """Initialize METEOR scorer."""
        self.scorer = Meteor()
    
    def compute(self, candidates, references):
        """
        Compute METEOR score.
        
        Args:
            candidates: {image_id: ["predicted caption"]}
            references: {image_id: ["ref1", "ref2", ...]}
            
        Returns:
            Dictionary with METEOR score (scaled to 0-100).
        """
        try:
            score, _ = self.scorer.compute_score(references, candidates)
            return {"METEOR": score * 100}
        except Exception as e:
            print(f"[Warning] METEOR computation failed: {e}")
            print("          METEOR requires NLTK data. Run: python -c \"import nltk; nltk.download('wordnet')\"")
            return {"METEOR": None}

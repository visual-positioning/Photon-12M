from pycocoevalcap.bleu.bleu import Bleu

class Metric:
    def __init__(self):
        # Bleu(4) computes BLEU-1, 2, 3, 4
        self.scorer = Bleu(4)

    def compute(self, candidates, references):
        """
        candidates: {img_id: ["cap"]}
        references: {img_id: ["ref1", "ref2"]}
        """
        score, _ = self.scorer.compute_score(references, candidates)
        
        # Return exact dictionary keys required by output
        return {
            "BLEU-1": score[0] * 100,
            "BLEU-2": score[1] * 100,
            "BLEU-3": score[2] * 100,
            "BLEU-4": score[3] * 100
        }
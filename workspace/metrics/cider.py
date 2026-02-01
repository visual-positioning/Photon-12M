from pycocoevalcap.cider.cider import Cider

class Metric:
    def __init__(self):
        self.scorer = Cider()

    def compute(self, candidates, references):
        score, _ = self.scorer.compute_score(references, candidates)
        return {"CIDEr": score * 100}
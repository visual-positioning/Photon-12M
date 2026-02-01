"""
SPICE Metric for Image Captioning Evaluation.

SPICE (Semantic Propositional Image Caption Evaluation) parses captions into
scene graphs (objects, attributes, relations) and compares the graphs using
F-score over semantic tuples.

Range: [0, 1] (multiplied by 100 for percentage)
Interpretation: Higher is better. Measures semantic correctness rather than
               surface-level word overlap. Ignores word order and grammar.

NOTE: SPICE requires Java to be installed. If Java is not available,
      this metric will gracefully skip and return None.
"""
import subprocess
import sys


class Metric:
    def __init__(self):
        """Initialize SPICE scorer."""
        self.scorer = None
        self.java_available = self._check_java()
        
        if self.java_available:
            try:
                from pycocoevalcap.spice.spice import Spice
                self.scorer = Spice()
            except Exception as e:
                print(f"[Warning] Could not initialize SPICE: {e}")
                self.scorer = None
    
    def _check_java(self):
        """Check if Java is available on the system."""
        try:
            result = subprocess.run(
                ['java', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def compute(self, candidates, references):
        """
        Compute SPICE score.
        
        Args:
            candidates: {image_id: ["predicted caption"]}
            references: {image_id: ["ref1", "ref2", ...]}
            
        Returns:
            Dictionary with SPICE score (scaled to 0-100), or None if unavailable.
        """
        if self.scorer is None:
            if not self.java_available:
                print("[Warning] SPICE requires Java. Please install Java runtime.")
            else:
                print("[Warning] SPICE scorer not initialized properly.")
            return {"SPICE": None}
        
        try:
            score, _ = self.scorer.compute_score(references, candidates)
            return {"SPICE": score * 100}
        except Exception as e:
            print(f"[Warning] SPICE computation failed: {e}")
            return {"SPICE": None}

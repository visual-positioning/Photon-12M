# Metric plugins for the evaluation framework
# Each module must implement a Metric class with:
#   - compute(candidates, references) -> Dict[str, float]
#     where:
#       candidates: {image_id: ["predicted caption"]}
#       references: {image_id: ["ref1", "ref2", ...]}

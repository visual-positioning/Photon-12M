# Method plugins for the evaluation framework
# Each module must implement a CaptionModel class with:
#   - __init__(device, checkpoint_path=None)
#   - generate_batch(List[PIL.Image]) -> List[str]
#   - generate(PIL.Image) -> str (optional fallback)

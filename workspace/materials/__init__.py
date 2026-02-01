# Dataset plugins for the evaluation framework
# Each module must implement a CaptionDataset class with:
#   - __getitem__(idx) -> {'image': PIL.Image, 'image_id': int, 'captions': List[str]}
#   - collate_fn(batch) -> batched dictionary

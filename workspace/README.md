# Image Captioning Evaluation Framework

A **plugin-based benchmarking framework** for evaluating image captioning models. Designed for zero-touch extensibility — add new datasets, methods, or metrics by simply dropping a Python file into the corresponding folder.

---

## 📁 Directory Structure

```
workspace/
├── evaluate.py              # Main orchestrator script
├── output.json              # Results log (auto-generated, appended)
│
├── common/                  # Shared utilities
│   ├── __init__.py
│   ├── registry.py          # Plugin discovery logic
│   └── io_utils.py          # Thread-safe JSON I/O
│
├── materials/               # Dataset plugins
│   ├── __init__.py
│   ├── coco2017.py          # COCO 2017 loader
│   ├── flickr30k.py         # Flickr30K loader
│   ├── nocaps.py            # NoCaps loader
│   └── visualgenome.py      # Visual Genome loader
│
├── methods/                 # Model plugins
│   ├── __init__.py
│   ├── blip.py              # BLIP (HuggingFace)
│   ├── oscar.py             # Microsoft GIT/OSCAR
│   ├── photon.py            # Photon (your model)
│   └── custom.py            # Ablation wrapper
│
└── metrics/                 # Metric plugins
    ├── __init__.py
    ├── bleu.py              # BLEU-1,2,3,4
    ├── cider.py             # CIDEr
    ├── rouge.py             # ROUGE-L
    ├── meteor.py            # METEOR
    └── spice.py             # SPICE (requires Java)
```

---

## 📂 Dataset Root Structure

The `--data_root` argument should point to a directory containing your datasets. Each dataset plugin expects its data in a specific folder structure:

```
<data_root>/
├── COCO2017/
│   ├── annotations/
│   │   ├── captions_train2017.json
│   │   └── captions_val2017.json
│   ├── train2017/
│   │   └── *.jpg
│   ├── val2017/
│   │   └── *.jpg
│   └── test2017/
│       └── *.jpg
│
├── FLICKR30K/
│   ├── images/
│   │   └── *.jpg
│   └── dataset_flickr30k.json  (Karpathy split)
│       OR results_20130124.token
│
├── NOCAPS/
│   ├── images/
│   │   └── val/*.jpg
│   └── nocaps_val_4500_captions.json
│
└── VISUALGENOME/
    ├── images/
    │   ├── VG_100K/*.jpg
    │   └── VG_100K_2/*.jpg
    └── region_descriptions.json
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision transformers pillow tqdm pycocoevalcap
```

For METEOR: `python -c "import nltk; nltk.download('wordnet')"`  
For SPICE: Install Java Runtime (JRE)

### Basic Usage

```bash
# Evaluate BLIP on COCO with CIDEr metric
python evaluate.py --dataset coco2017 --method blip --metric cider --data_root "C:\Dataset" --verbose

# Evaluate all methods on all datasets with all metrics
python evaluate.py --dataset all --method all --metric all --data_root "C:\Dataset"

# Ablation study with custom checkpoint
python evaluate.py --dataset coco2017 --method custom --checkpoint ./checkpoints/epoch_100.pth --metric bleu cider --remark "Epoch 100 test"
```

---

## ⚙️ Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset(s) to evaluate: `coco2017`, `flickr30k`, `nocaps`, `visualgenome`, or `all` | `all` |
| `--method` | Method(s) to test: `blip`, `oscar`, `photon`, `custom`, or `all` | `all` |
| `--metric` | Metric(s) to compute: `bleu`, `cider`, `rouge`, `meteor`, `spice`, or `all` | `all` |
| `--data_root` | **Required.** Path to dataset root directory | — |
| `--checkpoint` | Path to `.pth` file (required for `custom` method) | `None` |
| `--output` | Output JSON file path | `output.json` |
| `--batch_size` | Batch size for inference | `8` |
| `--device` | Device: `cuda` or `cpu` | `cuda` |
| `--experiment_id` | Unique ID for this run | Auto-generated |
| `--remark` | Notes/description for this experiment | `""` |
| `--verbose` | Print results to console | `False` |

---

## 🔌 Adding New Plugins

### Adding a New Dataset

1. Create `materials/my_dataset.py`
2. Implement the `CaptionDataset` class:

```python
from torch.utils.data import Dataset
from PIL import Image

class CaptionDataset(Dataset):
    def __init__(self, root, split='val', mode='full'):
        """
        Args:
            root: Path to dataset (e.g., <data_root>/MY_DATASET)
            split: 'train', 'val', or 'test'
            mode: 'full' (image + captions) or 'captions_only'
        """
        self.samples = []  # List of {image_id, file_name, captions}
        # Load your annotations here...
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        result = {
            'image_id': item['image_id'],
            'captions': item['captions']  # List[str]
        }
        if self.mode == 'full':
            result['image'] = Image.open(item['path']).convert('RGB')
        return result
    
    def collate_fn(self, batch):
        return {
            'image_id': [x['image_id'] for x in batch],
            'captions': [x['captions'] for x in batch],
            'image': [x['image'] for x in batch] if 'image' in batch[0] else None
        }
```

---

### Adding a New Method

1. Create `methods/my_model.py`
2. Implement the `CaptionModel` class:

```python
import torch
from PIL import Image
from typing import List

class CaptionModel:
    def __init__(self, device='cuda', checkpoint_path=None):
        """
        Args:
            device: 'cuda' or 'cpu'
            checkpoint_path: Optional path to model weights
        """
        self.device = device
        # Load your model here...
        # self.model = YourModel()
        # self.model.load_state_dict(torch.load(checkpoint_path))
        # self.model.to(device).eval()
    
    def generate_batch(self, image_list: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of PIL Images."""
        # Your inference logic here
        return ["caption for image 1", "caption for image 2", ...]
    
    def generate(self, image: Image.Image) -> str:
        """Generate caption for a single image (fallback)."""
        return self.generate_batch([image])[0]
```

---

### Adding a New Metric

1. Create `metrics/my_metric.py`
2. Implement the `Metric` class:

```python
class Metric:
    def __init__(self):
        # Initialize your scorer
        pass
    
    def compute(self, candidates, references):
        """
        Args:
            candidates: {image_id: ["predicted caption"]}
            references: {image_id: ["ref1", "ref2", ...]}
        
        Returns:
            Dict with metric scores, e.g., {"MyMetric": 85.5}
        """
        # Compute scores...
        return {"MyMetric": score * 100}
```

---

## 📊 Output Format

Results are appended to `output.json` as a list of experiment records:

```json
[
    {
        "experiment_id": "EXP_20260201_223933",
        "timestamp": "2026-02-01 22:39:33",
        "remark": "Baseline BLIP evaluation",
        "configuration": {
            "dataset": "coco2017",
            "method": "blip",
            "checkpoint": "default",
            "metrics_list": ["cider"]
        },
        "results": {
            "CIDEr": 113.4
        }
    }
]
```

---

## 📈 Metrics Reference

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **BLEU-1/2/3/4** | 0-100 | N-gram precision. BLEU-4 measures fluency. |
| **CIDEr** | 0-400+ | TF-IDF consensus. **Gold standard** for captioning. |
| **ROUGE-L** | 0-100 | Longest common subsequence. Measures structure. |
| **METEOR** | 0-100 | Includes synonyms/stemming. Better human correlation. |
| **SPICE** | 0-100 | Semantic scene graph matching. Ignores grammar. |

---

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `Path like C:Dataset` missing backslash | Use quotes: `--data_root "C:\Dataset"` |
| Memory allocation error | Reduce batch size: `--batch_size 4` or `--batch_size 1` |
| `No module named pycocoevalcap` | `pip install pycocoevalcap` |
| METEOR WordNet error | `python -c "import nltk; nltk.download('wordnet')"` |
| SPICE requires Java | Install Java Runtime Environment |

---

## 📜 License

Research use only. See main repository for license details.

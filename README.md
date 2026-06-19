# Explainable Deep Learning Framework for Multiclass Classification from MRI using Grad-CAM

A PyTorch project for multiclass brain tumor MRI classification using a dual-backbone
(ConvNeXt + Swin Transformer) fusion model, trained with the SAM optimizer, and explained
using Grad-CAM tumor localization heatmaps.

## Project Structure

```
mri-gradcam-classification/
├── src/
│   ├── __init__.py
│   ├── config.py              # dataset paths + training constants (TRAIN_DIR, TEST_DIR, EPOCHS, PATIENCE)
│   ├── data_visualization.py  # EDA plots: class distribution, sample images, pixel intensity, split visualization
│   ├── transforms.py          # Albumentations train/val transforms + augmentation visualization
│   ├── dataset.py             # MRIDataset, TransformSubset, get_loaders_with_val
│   ├── model.py                # GatedFusion, AstraNetBT (ConvNeXt + Swin fusion model), get_model
│   ├── optimizer.py           # SAM (Sharpness-Aware Minimization) optimizer
│   ├── engine.py              # train / evaluate / evaluate_loss loops
│   ├── visualize_results.py   # confusion matrix + training history plots
│   ├── gradcam.py             # Grad-CAM brain tumor localization visualization
│   └── main.py                # orchestrates the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## 1. Clone / Download

```bash
git clone <your-repo-url>
cd mri-gradcam-classification
```

## 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `pip install grad-cam` doesn't give you the exact API version used in this project,
install `pytorch-grad-cam` directly from GitHub instead:

```bash
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
```

## 4. Dataset

This project expects an MRI dataset split into `Training` and `Testing` folders, each
containing one subfolder per class, e.g.:

```
dataset1/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Open `src/config.py` and set `TRAIN_DIR` / `TEST_DIR` to the location of these folders
on your machine:

```python
TRAIN_DIR = "/path/to/dataset1/Training"
TEST_DIR  = "/path/to/dataset1/Testing"
```

## 5. Run

From the project root, run the full pipeline as a module:

```bash
python -m src.main
```

This single command runs everything in order:

1. Exploratory data analysis plots (class distribution, sample images, pixel intensity)
2. Augmentation visualization
3. Dataset/DataLoader construction (train/val/test split)
4. Model summary (AstraNetBT: ConvNeXt-Tiny + Swin-Tiny gated fusion)
5. Training loop with the SAM optimizer (20 epochs, early stopping patience 5),
   saving the best weights to `best_model.pth`
6. Training history plots (loss/accuracy curves)
7. Test-set evaluation (classification report + confusion matrix)
8. Grad-CAM brain tumor localization visualizations

## Running Individual Pieces

You can also import and use any module on its own, e.g. in a Python shell or your own script:

```bash
python3
```

```python
from src.config import TRAIN_DIR, TEST_DIR
from src.dataset import get_loaders_with_val
from src.model import AstraNetBT, get_model
from src.optimizer import SAM
from src.engine import train, evaluate
from src.gradcam import visualize_brain_tumors_localization
from src.visualize_results import plot_confusion_matrix, plot_history

train_loader, val_loader, test_loader, classes = get_loaders_with_val(
    TRAIN_DIR, TEST_DIR, batch_size=64, val_split=0.2
)
model = get_model(num_classes=4)
```

## Output

- `best_model.pth` — best model checkpoint saved during training (in the directory
  you ran the command from)
- Matplotlib windows/figures for every plot (class distribution, augmentation preview,
  dataset split, training curves, confusion matrix, Grad-CAM heatmaps)
- Printed `classification_report` (precision/recall/F1 per class) on the test set

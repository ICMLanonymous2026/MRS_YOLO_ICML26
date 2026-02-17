
# MRS-YOLO (ICLR 2026 Submission)

This repository contains the anonymized implementation of our ICLR 2026 submission.  
The code is provided for transparency and reproducibility in line with the double-blind review policy.  

---

## Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Minimal demos
We provide small synthetic demos so reviewers can verify that the code runs end-to-end without requiring access to the real dataset:

- `scripts/generate_dummy_data.py`  
  Creates a synthetic dataset (`data_dummy/`) with the same file structure and input format as in the experiments (spectrograms saved as `.npy` with corresponding JSON labels).

- `examples/minimal_forward.py`  
  Loads a dummy spectrogram and runs a single forward pass through the model, printing the output shapes.

- `examples/minimal_train.py`  
  Runs a one-epoch training loop on the dummy dataset.

```bash
python scripts/generate_dummy_data.py
python examples/minimal_forward.py
python examples/minimal_train.py
```

Running these demos reproduces the expected input/output interface and the key pipeline stages.

---

## Note about datasets

The real datasets used in the paper cannot be released due to confidentiality constraints.  
Instead, this repository provides dummy data generation and clear instructions to reproduce the expected structure of a usable dataset.  

Two dataset formats are supported:

### 1. Ultralytics-style (YOLO-txt)
```
data_dir/
 ├─ train/
 │   ├─ images/   # spectrograms saved as .pt
 │   └─ labels/   # YOLO-format .txt files (class xc yc w h)
 ├─ val/
 │   ├─ images/
 │   └─ labels/
 └─ data.yaml
```

Example training command:
```python
model.fit(
    data_dir="path/to/data_yolo",
    batch_size=16,
    dataset="ultralytics",
)
```

### 2. Fused JSON-style (multi-resolution)
```
data_dir/
 ├─ train/
 │   ├─ images/   # spectrograms as .pt lists [Tensor(C,H,W), ...]
 │   └─ labels/   # JSON files with bounding boxes + metadata (class, xc, yc, w, h, SNR, PSNR at multiple resolutions)
 ├─ val/
 │   ├─ images/
 │   └─ labels/
```

Example JSON label entry:
```json
{
  "labels": [
    {
      "class": 14,
      "xc": 0.41,
      "yc": 0.52,
      "w": 0.71,
      "h": 0.74,
      "snr": -11.6,
      "psnr": {
        "cfg128": 6.32,
        "cfg256": 8.60,
        "cfg512": 8.87
      }
    }
  ]
}
```

Example training command:
```python
model.fit(
    data_dir="path/to/yolo_fused_multires_dataset",
    batch_size=64,
    dataset="fused",
    epochs=100,
    patience=10,
)
```

---

## Reproducibility

- **Training protocol, hyperparameters, and compute details** are fully described in the paper.  
- The code here provides the model definitions, training loop, and dataset interface.  
- Synthetic demos validate that the pipeline executes correctly; real results are obtained only with the confidential datasets described in the manuscript.  
- Additional artifacts (logs, checkpoints, further documentation) will be released after the review phase, in line with conference policy.  

---

*This repository is anonymized for double-blind review. Do not attempt to infer author identity.*

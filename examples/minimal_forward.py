"""
Minimal forward pass demo (ICLR)

Prereq:
    python scripts/generate_dummy_data.py

Run:
    python examples/minimal_forward.py
"""

import os
import json
import numpy as np
import torch


from ..mrs_yolo.models.mr_yolo import MRSYOLOModel as Model 



def load_first_dummy(spec_dir="data_dummy"):
    """Load the first synthetic spectrogram saved by scripts/generate_dummy_data.py."""
    meta_path = os.path.join(spec_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    npy_path = meta[0]["file"]  # first sample path (e.g., data_dummy/spec_0000.npy)
    x = np.load(npy_path).astype(np.float32)  # shape (C,H,W) = (1,512,512)
    return x  # numpy array (1,512,512)


def main():
    # ---- Data ----
    x = load_first_dummy("data_dummy")             # (1, 512, 512)
    x_t = torch.from_numpy(x).unsqueeze(0)         # (B=1, C=1, H=512, W=512)

    # ---- Device ----
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_t = x_t.to(device)

    # ---- Model ----
    # Single-resolution example matching (1,512,512)
    input_resolutions = [(512, 512)]

    model = Model(
        num_classes=15,
        device=device,
        reg_max=16,
        output_dir="runs/minimal_forward",
        input_resolutions=input_resolutions,
        width_mult=0.5,
        backbone_mode="pyramid",
        outfusion_channels_mult=1,
    ).to(device)

    model.eval()

    preds, _, _ = model.predict(
        x_t,
        to_plot="output/visu4.png",
        conf_threshold=0.6,
    )

    def brief(obj):
        if isinstance(obj, torch.Tensor):
            return f"Tensor{tuple(obj.shape)} dtype={obj.dtype}"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}[{len(obj)}]"
        if isinstance(obj, dict):
            keys = ", ".join(list(obj.keys())[:6])
            return f"dict(keys=[{keys}])"
        return str(type(obj))

    if isinstance(preds, (list, tuple)):
        print("[OK] Forward done. Output is a", type(preds).__name__)
        for i, p in enumerate(preds[:4]):
            print(f"  [{i}] -> {brief(p)}")
    elif isinstance(preds, dict):
        print("[OK] Forward done. Output dict with keys:", list(preds.keys())[:10])
        for k, v in list(preds.items())[:4]:
            print(f"  {k}: {brief(v)}")
    else:
        print("[OK] Forward done. Output:", brief(preds))


if __name__ == "__main__":
    main()

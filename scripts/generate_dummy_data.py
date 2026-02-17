# scripts/generate_dummy_data.py
import os
import numpy as np
import torch
from PIL import Image
import json

OUT = "data_dummy"
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

def save_spectrogram(i, shape=(1,512,512)):
    x = (np.random.randn(*shape) * 0.1 + 0.5).astype(np.float32)
    fname = os.path.join(OUT, f"spec_{i:04d}.npy")
    np.save(fname, x)
    return fname

meta = []
for i in range(20):
    f = save_spectrogram(i, shape=(1,512,512))
    # one dummy box [x,y,w,h,class,score]
    boxes = [{"bbox":[50,60,30,10],"label":0}]
    meta.append({"file":f, "boxes":boxes})

with open(os.path.join(OUT,"metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print("Dummy data written to", OUT)

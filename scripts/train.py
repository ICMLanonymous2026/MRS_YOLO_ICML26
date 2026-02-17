#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICLR submission — minimal training entrypoint.

- Aucune dépendance à des chemins locaux.
- Déduction optionnelle des résolutions d'entrée depuis un .pt d'exemple.
- Arguments clairs (data, epochs, batch, device, etc.).
- Compatible single- ou multi-resolution (list[(H,W)]).
- Ne suppose pas l'existence d'un dataset/éval dans le repo.

Exemples d'usage :
  python scripts/train.py --data /path/to/dataset --epochs 100
  python scripts/train.py --data /path/to/dataset --detect-res --batch 32 --imgsz 512
  python scripts/train.py --data /path/to/dataset --res 256x256 512x256 128x512
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch

# ⚠️ Adapte l'import ci-dessous à ton package final :
# - si tu gardes "mrs_yolo", décommente la ligne 33
# - si tu conserves "yolo_perso", décommente la ligne 34
# from mrs_yolo.models.mr_yolo import MRSYOLOModel as Model
from yolo_perso.models.mr_yolo import MR_YOLO as Model  # ← provisoire si ton code n'est pas encore migré

# -----------------------------------------------------------------------------


def parse_resolutions(tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Parse des résolutions du style: 256x256 512x256 128x512 -> [(256,256), (512,256), (128,512)]
    """
    out = []
    for t in tokens:
        t = t.lower().strip().replace(" ", "")
        if "x" not in t:
            raise ValueError(f"Bad resolution token '{t}'. Expected format HxW (e.g., 256x256).")
        h, w = t.split("x")
        out.append((int(h), int(w)))
    return out


def detect_input_resolutions(data_dir: str, split: str = "train") -> List[Tuple[int, int]]:
    """
    Cherche un .pt exemple dans data_dir/<split>/images, le charge, et infère les résolutions.
    - Si le .pt contient une LISTE de Tensors -> multi-résolution.
    - Si c'est un unique Tensor -> single-résolution.
    """
    images_dir = Path(data_dir) / split / "images"
    example_pt = next(images_dir.glob("*.pt"), None)
    if example_pt is None:
        raise FileNotFoundError(f"No .pt file found in {images_dir}. Provide --res or place sample .pt.")

    specs = torch.load(example_pt, map_location="cpu")
    resolutions: List[Tuple[int, int]] = []

    if isinstance(specs, list):
        # liste de tenseurs (C,H,W) ou (H,W)
        for i, spec in enumerate(specs):
            if not torch.is_tensor(spec):
                raise ValueError(f"Element {i} in {example_pt} is {type(spec)}, expected Tensor.")
            if spec.ndim == 3:
                _, H, W = spec.shape
            elif spec.ndim == 2:
                H, W = spec.shape
            else:
                raise ValueError(f"Unexpected ndim={spec.ndim} for element {i} in {example_pt}")
            resolutions.append((H, W))
    elif torch.is_tensor(specs):
        if specs.ndim == 3:
            _, H, W = specs.shape
        elif specs.ndim == 2:
            H, W = specs.shape
        else:
            raise ValueError(f"Unexpected ndim={specs.ndim} for {example_pt}")
        resolutions.append((H, W))
    else:
        raise ValueError(f"Expected list[Tensor] or Tensor in {example_pt}, got {type(specs)}")

    return resolutions


def main():
    p = argparse.ArgumentParser(description="ICLR minimal training launcher")
    p.add_argument("--data", required=True, help="Path to dataset root (expects <split>/images/*.pt layout)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=512, help="Optional unified training size if your loader resizes")
    p.add_argument("--device", default="auto", help='"auto", "cpu", or "cuda:0" ...')
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-dir", default="runs/exp0")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience if supported by your trainer")
    p.add_argument("--reg-max", type=int, default=16, help="Distribution Focal Loss bins (if applicable)")
    p.add_argument("--classes", type=int, default=15, help="Number of classes")
    p.add_argument("--width-mult", type=float, default=0.5, help="Backbone width multiplier")
    p.add_argument("--outfusion-ch-mult", type=float, default=1.0, help="Channels multiplier after fusion")
    p.add_argument("--backbone-mode", default="pyramid", choices=["pyramid", "late-fusion", "max-upsample"])
    p.add_argument("--dataset-type", default="dataset512", help="Adapter/flag used by your Dataset implementation")
    p.add_argument("--detect-res", action="store_true", help="Auto-detect input resolutions from a sample .pt")
    p.add_argument("--res", nargs="+", help='Manual list of resolutions: e.g., --res 256x256 512x256')

    args = p.parse_args()

    # Résolutions d'entrée : détection auto ou manuelle, sinon fallback single-res imgsz x imgsz.
    if args.detect_res:
        input_resolutions = detect_input_resolutions(args.data, split="train")
    elif args.res:
        input_resolutions = parse_resolutions(args.res)
    else:
        input_resolutions = [(args.imgsz, args.imgsz)]

    print(f"[info] input_resolutions = {input_resolutions}")

    # Choix du device
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[info] device = {device}")

    # Instanciation du modèle
    # ⚠️ Adapte les kwargs en fonction de ta signature réelle de constructeur.
    model = Model(
        num_classes=args.classes,
        device=device,
        reg_max=args.reg_max,
        output_dir=args.save_dir,
        input_resolutions=input_resolutions,
        width_mult=args.width_mult,
        backbone_mode=args.backbone_mode,
        outfusion_channels_mult=args.outfusion_ch_mult,
    )

    # Entraînement
    # ⚠️ Adapte les kwargs de fit(...) à ton implémentation (noms/flags).
    model.fit(
        data_dir=args.data,
        batch_size=args.batch,
        dataset=args.dataset_type,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,           # si ton loader en tient compte
        num_workers=args.num_workers
    )

    print(f"[done] Training finished. Artifacts (if any) saved under: {args.save_dir}")


if __name__ == "__main__":
    # chmod +x scripts/train.py
    main()

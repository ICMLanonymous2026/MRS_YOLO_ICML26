import os, re, glob, json
from pathlib import Path
from typing import List, Dict, Tuple, Any 
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np

_NUMERIC_RE = re.compile(r"(\d+)$")

class YOLODatasetFusedMultiRes(Dataset):
    def __init__(self, data_dir, labels_dir, res_keys=["cfg128","cfg256","cfg512","cfg1024","cfg2048"], max_dim=1024):
        """
        Args:
            data_dir: directory of .pt files, each is list[Tensor] (C,H,W), one per resolution
            labels_dir: directory with matching .json files
            res_keys: list of PSNR keys in the exact order of your multi-res inputs,
                      e.g. ["cfg128","cfg256","cfg512","cfg1024","cfg2048"].
                      If None, we'll infer keys from the first JSON we find and sort numerically.
        """
        self.data_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        ])
        self.labels_dir = labels_dir
        self._res_keys = res_keys  
        self.max_dim = max_dim

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _numeric_key(k):
        m = _NUMERIC_RE.search(k)
        return int(m.group(1)) if m else float("inf")

    def _ensure_res_keys(self, psnr_dict):
        """Infer and lock resolution key order if not provided."""
        if self._res_keys is not None:
            return
        if not isinstance(psnr_dict, dict) or not psnr_dict:
            return
        keys = list(psnr_dict.keys())
        keys.sort(key=self._numeric_key)
        self._res_keys = keys

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        specs = torch.load(data_path, map_location="cpu")  
        
        if not isinstance(specs, list):
            raise ValueError(f"Expected a list of Tensors, got {type(specs)}")
        

        def to_chw(t):
            if t.ndim == 2: return t.unsqueeze(0)
            if t.ndim == 3: return t
            raise ValueError(f"Unexpected spec shape: {t.shape}")
        
        specs = [to_chw(t) for t in specs if t.shape[-2] <= self.max_dim and t.shape[-1] <= self.max_dim]
            
        base = os.path.splitext(os.path.basename(data_path))[0]
        json_path = os.path.join(self.labels_dir, base + ".json")

        cls, bboxes, snrs = [], [], []
        psnrs_per_obj = []  

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            if self._res_keys is None:
                for item in data.get("labels", []):
                    if isinstance(item.get("psnr", None), dict) and item["psnr"]:
                        self._ensure_res_keys(item["psnr"])
                        break

            for item in data.get("labels", []):
                cls.append(item["class"])
                bboxes.append([item["xc"], item["yc"], item["w"], item["h"]])
                snr_val = item.get("snr", None)
                snrs.append(float(snr_val) if snr_val is not None else -1.0)

                # PSNR vector aligned to res_keys; fill -1.0 if missing/unavailable
                vec = []
                if self._res_keys is None:
                    # No keys known → try to read dict order (not guaranteed) or set empty
                    if isinstance(item.get("psnr", None), dict) and item["psnr"]:
                        # sort keys on the fly to keep consistency within this sample
                        local_keys = sorted(item["psnr"].keys(), key=self._numeric_key)
                        vec = [float(item["psnr"].get(k, -1.0)) for k in local_keys]
                        # lock keys for the rest of the dataset run
                        self._res_keys = local_keys
                    else:
                        vec = []
                else:
                    for k in self._res_keys:
                        v = item.get("psnr", {}).get(k, None)
                        vec.append(float(v) if v is not None else -1.0)

                psnrs_per_obj.append(vec)

        # To tensors
        cls    = torch.tensor(cls,    dtype=torch.float32)
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0,4), dtype=torch.float32)
        snrs   = torch.tensor(snrs,   dtype=torch.float32)

        # PSNR tensor (N, R) or (0, R)
        R = len(self._res_keys) if self._res_keys is not None else 0
        if len(psnrs_per_obj) == 0:
            psnr_tensor = torch.zeros((0, R), dtype=torch.float32)
        else:
            # ensure every row has length R
            if R == 0:
                # no keys known but we have psnr data — infer R from first row
                R = len(psnrs_per_obj[0]) if psnrs_per_obj else 0
                self._res_keys = [f"cfg{i}" for i in range(R)]  # placeholder names
            psnr_tensor = torch.tensor(psnrs_per_obj, dtype=torch.float32)
            if psnr_tensor.ndim != 2 or psnr_tensor.shape[1] != R:
                # Pad/crop defensively to (N,R)
                N = psnr_tensor.shape[0]
                fixed = torch.full((N, R), -1.0, dtype=torch.float32)
                cols = min(R, psnr_tensor.shape[1])
                if cols > 0:
                    fixed[:, :cols] = psnr_tensor[:, :cols]
                psnr_tensor = fixed

        return {
            "imgs":    specs,         # LIST of Tensors [(C,H,W), …]
            "cls":     cls,           # (N,)
            "bboxes":  bboxes,        # (N,4)
            "snr":     snrs,          # (N,)
            "psnr":    psnr_tensor,   # (N,R)
            "img_idx": idx,
            "res_keys": self._res_keys,  # helpful for downstream alignment/debug
        }

    @staticmethod
    def collate_fn(batch):
        # images: transpose lists, stack per resolution -> List[Tensor] len=R, each (B,C,Hi,Wi)
        imgs_lists = [item["imgs"] for item in batch]
        imgs_per_res = list(zip(*imgs_lists))
        imgs = [torch.stack(res_list, dim=0) for res_list in imgs_per_res]

        # targets parts
        all_cls    = [item["cls"]     for item in batch]
        all_boxes  = [item["bboxes"]  for item in batch]
        all_snrs   = [item["snr"]     for item in batch]
        all_psnrs  = [item["psnr"]    for item in batch]

        targets = []
        for i, (cls, boxes, snr, psnr) in enumerate(zip(all_cls, all_boxes, all_snrs, all_psnrs)):
            if boxes.numel():
                img_idx = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
                cls_col = cls.unsqueeze(-1)
                snr_col = snr.unsqueeze(-1)
                # Concatenate: [img_idx, cls, x,y,w,h, snr, psnr...]
                row = torch.cat((img_idx, cls_col, boxes, snr_col, psnr), dim=1)
                targets.append(row)

        targets = torch.cat(targets, 0) if targets else torch.zeros((0, 7 + (all_psnrs[0].shape[1] if all_psnrs else 0)), dtype=torch.float32)

        res_keys = None
        for item in batch:
            if item.get("res_keys"):
                res_keys = item["res_keys"]
                break

        return imgs, targets, res_keys


class YOLODatasetSplitMultiRes(Dataset):
    """
    Un sample = un unique fichier sc_<stem>_cfg<res>.pt
    - __init__: repère tous les .pt et extrait (stem, cfg_res)
    - __getitem__: charge ce fichier, le pad à (T,T), 
                   charge stem.json et adapte **ses** labels
    - collate_fn: stack des images (B,C,T,T) & concat targets
    """
    def __init__(self,
                 data_dir: str,
                 labels_dir: str,
                 target_length: int = 1024):
        self.data_paths = sorted(Path(data_dir).glob("sc_*_cfg*.pt"))
        if not self.data_paths:
            raise FileNotFoundError(f"No PT files in {data_dir}")
        self.labels_dir = Path(labels_dir)
        self.target_len = target_length

        self._rx = re.compile(r"^(sc_[A-Za-z0-9]+)_cfg(\d+)$")

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> dict:
        data_path = self.data_paths[idx]
        name      = data_path.stem
        m = self._rx.match(name)
        if not m:
            raise ValueError(f"Filename {name} doesn't match pattern")
        stem, res = m.group(1), int(m.group(2))

        # 1) load spectrogram and ensure shape (C,H,W)
        spec = torch.load(data_path, map_location="cpu")
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        elif spec.ndim != 3:
            raise ValueError(f"Expected ndim 2 or 3, got {spec.ndim}")
        C, H, W = spec.shape

        # 2) pad/crop symétrique to (C, T, T)
        dh = self.target_len - H
        dw = self.target_len - W
        top, bottom = dh//2, dh - dh//2
        left, right = dw//2, dw - dw//2
        img = F.pad(spec, (left, right, top, bottom), value=0)

        # 3) load labels JSON once, then filter only this res?
        json_path = self.labels_dir / f"{stem}.json"
        cls, bboxes, snrs = [], [], []
        if json_path.exists():
            data = json.loads(json_path.read_text())
            for item in data["labels"]:
                # chaque label reste valable sur **toutes** les configs
                xc0, yc0 = item["xc"], item["yc"]
                w0,  h0  = item["w"],  item["h"]
                snr_val  = item.get("snr", -1.0)

                # recalcule boxes pour **cette** image
                x_abs = xc0 * W + left
                y_abs = yc0 * H + top
                w_abs = w0  * W
                h_abs = h0  * H

                xc_n = x_abs / self.target_len
                yc_n = y_abs / self.target_len
                w_n  = w_abs / self.target_len
                h_n  = h_abs / self.target_len

                cls.append(item["class"])
                bboxes.append([xc_n, yc_n, w_n, h_n])
                snrs.append(snr_val)

        # 4) build targets Tensor: img_idx always zero (seul sample)
        if bboxes:
            boxes   = torch.tensor(bboxes, dtype=torch.float32)
            classes = torch.tensor(cls,     dtype=torch.float32).unsqueeze(1)
            snr_t   = torch.tensor(snrs,    dtype=torch.float32).unsqueeze(1)
            img_idx = torch.zeros((boxes.shape[0],1), dtype=torch.float32)
            targets = torch.cat([img_idx, classes, boxes, snr_t], dim=1)
        else:
            targets = torch.zeros((0,7), dtype=torch.float32)

        return {"img": img, "targets": targets}

    @staticmethod
    def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - imgs: Tensor (B, C, T, T)
        - targets: Tensor (N,7), targets[:,0] = image index in [0..B-1]
        """
        imgs = torch.stack([b["img"] for b in batch], dim=0)
        all_tgts = []
        for i, b in enumerate(batch):
            t = b["targets"]
            if t.numel():
                t[:,0] = i
                all_tgts.append(t)
        targets = torch.cat(all_tgts, dim=0) if all_tgts else torch.zeros((0,7))
        return imgs, targets
    

class YOLODatasetSplitMultiResAllConfigs(Dataset):
    """
    Un sample = toutes les résolutions d’un stem (sc_<stem>_cfg*.pt)
    - __init__: regroupe tous les fichiers par stem
    - __getitem__: charge les specs multi-résolutions + adapte les labels
    - collate_fn: regroupe imgs = list[Tensor (B,C,T,T)], gt_* = lists per-res
    """

    def __init__(self,
                 data_dir: str,
                 labels_dir: str,
                 target_length: int = 1024):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.target_len = target_length

        # 1. Regrouper tous les fichiers .pt par stem
        all_paths = sorted(self.data_dir.glob("sc_*_cfg*.pt"))
        self.samples: Dict[str, List[Tuple[int, Path]]] = {}

        rx = re.compile(r"^(sc_[a-f0-9]+)_cfg(\d+)$")
        for path in all_paths:
            m = rx.match(path.stem)
            if not m:
                continue
            stem, res = m.group(1), int(m.group(2))
            if stem not in self.samples:
                self.samples[stem] = []
            self.samples[stem].append((res, path))

        # 2. Liste des stems uniques
        self.stems = sorted(self.samples.keys())

    @staticmethod
    def _cfg_key(res: int) -> str:
        return f"cfg{res}"

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        entries = self.samples[stem]  # List of (res, path)
        entries = sorted(entries)     # ordre croissant de résolutions
        specs, res_list = [], []

        # Charger les labels une seule fois
        json_path = self.labels_dir / f"{stem}.json"
        labels = []
        if json_path.exists():
            data = json.loads(json_path.read_text())
            for item in data.get("labels", []):
                labels.append({
                    "class": item["class"],
                    "xc": item["xc"],
                    "yc": item["yc"],
                    "w": item["w"],
                    "h": item["h"],
                    "snr": item.get("snr", -1.0),
                    "psnr": item.get("psnr", {})  # dict potentiellement vide
                })

        gt_boxes_list = []
        gt_labels_list = []
        gt_snrs_list = []
        gt_psnrs_per_res_list = []  # List per-resolution: each is Tensor (N,)
        orig_hw_list = []  # (H_orig, W_orig)

        # On prépare aussi res_keys (ex. ["cfg128","cfg256",...]) pour debug
        res_keys = [self._cfg_key(res) for res, _ in entries]

        # 1) Charger specs et remplir meta
        for res, path in entries:
            spec = torch.load(path, map_location="cpu")
            if spec.ndim == 2:
                spec = spec.unsqueeze(0)
            elif spec.ndim != 3:
                raise ValueError(f"Unexpected spec ndim: {spec.ndim}")
            C, H, W = spec.shape
            orig_hw_list.append((H, W))

            # Padding symétrique en (C, T, T)
            dh = self.target_len - H
            dw = self.target_len - W
            top, bottom = dh // 2, dh - dh // 2
            left, right = dw // 2, dw - dw // 2
            img = F.pad(spec, (left, right, top, bottom), value=0)

            specs.append(img)
            res_list.append(res)

            # 2) Construire GT alignés (relatifs) + SNR + PSNR (pour CETTE résolution)
            boxes, classes, snrs, psnrs_this_res = [], [], [], []
            cfgk = self._cfg_key(res)

            for item in labels:
                # labels déjà relatifs → on garde tel quel
                boxes.append([item["xc"], item["yc"], item["w"], item["h"]])
                classes.append(item["class"])
                snrs.append(item["snr"])

                # PSNR pour cette résolution (ou -1.0 si absent)
                ps_val = item.get("psnr", {}).get(cfgk, None)
                psnrs_this_res.append(float(ps_val) if ps_val is not None else -1.0)

            if boxes:
                gt_boxes_list.append(torch.tensor(boxes, dtype=torch.float32))   # (N,4)
                gt_labels_list.append(torch.tensor(classes, dtype=torch.long))   # (N,)
                gt_snrs_list.append(torch.tensor(snrs, dtype=torch.float32))     # (N,)
                gt_psnrs_per_res_list.append(torch.tensor(psnrs_this_res, dtype=torch.float32))  # (N,)
            else:
                gt_boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
                gt_labels_list.append(torch.zeros((0,), dtype=torch.long))
                gt_snrs_list.append(torch.zeros((0,), dtype=torch.float32))
                gt_psnrs_per_res_list.append(torch.zeros((0,), dtype=torch.float32))

        return {
            "imgs": specs,                    # List[Tensor CxT×T]
            "gt_boxes": gt_boxes_list,        # List[Tensor Nx4] RELATIF (par rés)
            "gt_labels": gt_labels_list,      # List[Tensor N]    (par rés)
            "gt_snrs": gt_snrs_list,          # List[Tensor N]    (par rés)
            "gt_psnrs": gt_psnrs_per_res_list,# List[Tensor N]    (par rés)  ⬅️ NEW
            "orig_hw": orig_hw_list,          # List[(H,W)] par rés
            "res_keys": res_keys,             # ex. ["cfg128","cfg256",...]
        }

    @staticmethod
    def collate_fn(batch: List[dict]):
        # imgs: list[res][B,C,T,T]
        batch_imgs = [item["imgs"] for item in batch]
        imgs_per_res = list(zip(*batch_imgs))
        imgs = [torch.stack(img_list, dim=0) for img_list in imgs_per_res]

        def collate_list(key: str):
            raw = [item[key] for item in batch]   # list over B of (list per res)
            per_res = list(zip(*raw))            # tuple per res, each tuple has length B
            return [list(elem) for elem in per_res]

        gt_boxes = collate_list("gt_boxes")
        gt_labels = collate_list("gt_labels")
        gt_snrs = collate_list("gt_snrs")
        gt_psnrs = collate_list("gt_psnrs")   # ⬅️ NEW
        orig_hw  = collate_list("orig_hw")

        # res_keys: assume identiques dans le batch (même config de données)
        res_keys = None
        for item in batch:
            if "res_keys" in item and item["res_keys"]:
                res_keys = item["res_keys"]
                break

        return imgs, gt_boxes, gt_labels, gt_snrs, gt_psnrs, orig_hw, res_keys


class YOLODatasetSTFT512(Dataset):
    """
    Single-resolution dataset for STFT=512 (spectrograms expected to be 256x256).
    Returns a single Tensor in `specs` (C,H,W) instead of a list.
    """
    def __init__(self, data_dir, labels_dir, psnr_key="cfg512"):
        """
        Args:
            data_dir: directory of .pt files; each file should contain one Tensor (C,H,W)
                      for the 512-STFT spectrogram. If a list/dict is encountered,
                      the 256x256 entry will be selected automatically.
            labels_dir: directory with matching .json files
            psnr_key: key to read PSNR from labels when available (default: "cfg512")
        """
        self.data_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        ])
        self.labels_dir = labels_dir
        self.psnr_key = psnr_key  # e.g., "cfg512"

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _pick_256x256(entry):
        """
        Robustly extract a (C,256,256) tensor from various saved formats:
        - Tensor: (C,256,256) or (256,256) -> ensure channel dim
        - list/tuple: pick first item with H=W=256
        - dict: try key 'cfg512' else first item with H=W=256
        """
        def _ensure_chw(x):
            if not isinstance(x, torch.Tensor):
                return None
            if x.ndim == 2:  # (H,W)
                x = x.unsqueeze(0)
            if x.ndim == 3 and x.shape[-2:] == (256, 256):
                return x
            return None

        if isinstance(entry, torch.Tensor):
            x = _ensure_chw(entry)
            if x is not None: return x

        if isinstance(entry, (list, tuple)):
            for it in entry:
                x = _ensure_chw(it)
                if x is not None: return x

        if isinstance(entry, dict):
            # prefer explicit key if present
            if "cfg512" in entry:
                x = _ensure_chw(entry["cfg512"])
                if x is not None: return x
            # otherwise search any 256x256
            for it in entry.values():
                x = _ensure_chw(it)
                if x is not None: return x

        raise ValueError("Could not find a (C,256,256) spectrogram in the loaded .pt")

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        loaded = torch.load(data_path, map_location="cpu")
        spec = self._pick_256x256(loaded)  # Tensor (C,256,256)

        # --- Labels
        base = os.path.splitext(os.path.basename(data_path))[0]
        json_path = os.path.join(self.labels_dir, base + ".json")

        cls, bboxes, snrs, psnrs = [], [], [], []

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

            for item in data.get("labels", []):
                cls.append(item["class"])
                bboxes.append([item["xc"], item["yc"], item["w"], item["h"]])
                snr_val = item.get("snr", None)
                snrs.append(float(snr_val) if snr_val is not None else -1.0)

                # Single-resolution PSNR: read from dict if available
                v = None
                if isinstance(item.get("psnr", None), dict):
                    v = item["psnr"].get(self.psnr_key, None)
                elif isinstance(item.get("psnr", None), (int, float)):
                    v = item["psnr"]
                psnrs.append(float(v) if v is not None else -1.0)

        # --- To tensors
        cls_t    = torch.tensor(cls,    dtype=torch.float32)
        bboxes_t = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0,4), dtype=torch.float32)
        snr_t    = torch.tensor(snrs,   dtype=torch.float32)
        # keep (N,1) to mirror the multi-res shape (N,R)
        psnr_t   = torch.tensor(psnrs,  dtype=torch.float32).unsqueeze(1) if psnrs else torch.zeros((0,1), dtype=torch.float32)

        return {
            "specs":   spec,       # Tensor (C,256,256) — single spectrogram
            "cls":     cls_t,      # (N,)
            "bboxes":  bboxes_t,   # (N,4)
            "snr":     snr_t,      # (N,)
            "psnr":    psnr_t,     # (N,1) — single-resolution PSNR
            "img_idx": idx,
            "res_key": self.psnr_key,  # "cfg512"
        }
    
    @staticmethod
    def collate_fn(batch):
        # --- Images: stack single-resolution specs -> Tensor (B,C,256,256)
        imgs = torch.stack([item["specs"] for item in batch], dim=0)

        # --- Targets parts
        all_cls   = [item["cls"]    for item in batch]     # list of (N_i,)
        all_boxes = [item["bboxes"] for item in batch]     # list of (N_i,4)
        all_snrs  = [item["snr"]    for item in batch]     # list of (N_i,)
        all_psnrs = [item["psnr"]   for item in batch]     # list of (N_i,1)

        targets = []
        for i, (cls, boxes, snr, psnr) in enumerate(zip(all_cls, all_boxes, all_snrs, all_psnrs)):
            if boxes.numel():
                img_idx = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
                cls_col = cls.unsqueeze(-1)        # (N_i,1)
                snr_col = snr.unsqueeze(-1)        # (N_i,1)
                # Concatenate per object: [img_idx, cls, x, y, w, h, snr, psnr]
                row = torch.cat((img_idx, cls_col, boxes, snr_col, psnr), dim=1)  # (N_i, 8)
                targets.append(row)

        if targets:
            targets = torch.cat(targets, dim=0)  # (sum N_i, 8)
        else:
            # 7 base cols (img_idx, cls, x,y,w,h, snr) + 1 psnr = 8
            targets = torch.zeros((0, 8), dtype=torch.float32)

        # --- Carry the single res_key forward (assume same across batch)
        res_key = None
        for item in batch:
            if "res_key" in item and item["res_key"] is not None:
                res_key = item["res_key"]
                break

        return [imgs], targets, batch[0].get("res_key", None)



class YoloPTDataset(Dataset):
    """
    Lit des spectrogrammes .pt (H,W) ou (C,H,W) et des labels YOLO txt.
    - images_dir: dossier contenant directement les .pt (ex: .../train/images)
    - labels_dir: dossier contenant directement les .txt (ex: .../train/labels)
    - target_size: côté T de la sortie carrée (pad/crop centré)
    - pad_value: valeur de remplissage pour le padding
    Sortie __getitem__:
       dict(img: Tensor[C,T,T],
            targets: Tensor[N,8] = [img_idx, cls, cx, cy, w, h, snr, psnr])
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        target_size: int = 1024,
        pad_value: float = 0.0,
        snr_fill: float = 0.0,   # <-- valeurs de remplissage
        psnr_fill: float = -1.0, # <-- valeurs de remplissage
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        self.paths = sorted(self.images_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files in {self.images_dir}")

        self.T = int(target_size)
        self.pad_value = float(pad_value)
        self.snr_fill = float(snr_fill)
        self.psnr_fill = float(psnr_fill)

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _center_pad_crop_2d(x: torch.Tensor, out_h: int, out_w: int, pad_value: float) -> Tuple[torch.Tensor, int, int, int, int]:
        _, H, W = x.shape
        dh = out_h - H
        dw = out_w - W

        top_pad    = max(dh // 2, 0)
        bottom_pad = max(dh - dh // 2, 0)
        left_pad   = max(dw // 2, 0)
        right_pad  = max(dw - dw // 2, 0)

        y = x
        if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
            y = F.pad(y, (left_pad, right_pad, top_pad, bottom_pad), value=pad_value)

        crop_top  = max((-dh) // 2, 0)
        crop_left = max((-dw) // 2, 0)
        crop_bottom = crop_top + out_h
        crop_right  = crop_left + out_w
        if crop_top > 0 or crop_left > 0:
            y = y[:, crop_top:crop_bottom, crop_left:crop_right]

        return y, top_pad, left_pad, crop_top, crop_left

    def _read_labels_txt(self, stem: str) -> List[List[float]]:
        txt_path = self.labels_dir / f"{stem}.txt"
        if not txt_path.exists():
            return []
        lines = txt_path.read_text().strip().splitlines()
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = parts
            out.append([float(c), float(cx), float(cy), float(w), float(h)])
        return out

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        stem = p.stem

        # 1) Load tensor -> (C,H,W)
        arr = torch.load(p, map_location="cpu")
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        elif arr.ndim != 3:
            raise ValueError(f"{p.name}: expected 2D or 3D tensor, got shape {tuple(arr.shape)}")

        _, H, W = arr.shape

        # 2) Center pad/crop to (T,T)
        img, top_pad, left_pad, crop_top, crop_left = self._center_pad_crop_2d(arr, self.T, self.T, self.pad_value)

        # 3) Read labels and reproject boxes
        raw_labels = self._read_labels_txt(stem)
        targets_list = []
        for c, cx_n, cy_n, w_n, h_n in raw_labels:
            cx_abs = cx_n * W
            cy_abs = cy_n * H
            w_abs  = w_n  * W
            h_abs  = h_n  * H

            cx_adj = cx_abs + left_pad - crop_left
            cy_adj = cy_abs + top_pad  - crop_top

            x1 = cx_adj - w_abs/2
            y1 = cy_adj - h_abs/2
            x2 = cx_adj + w_abs/2
            y2 = cy_adj + h_abs/2

            x1 = max(0.0, min(float(self.T), float(x1)))
            y1 = max(0.0, min(float(self.T), float(y1)))
            x2 = max(0.0, min(float(self.T), float(x2)))
            y2 = max(0.0, min(float(self.T), float(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            cx_new = ((x1 + x2) / 2.0) / self.T
            cy_new = ((y1 + y2) / 2.0) / self.T
            w_new  = (x2 - x1) / self.T
            h_new  = (y2 - y1) / self.T

            # [img_idx, cls, cx, cy, w, h, snr, psnr]
            targets_list.append([
                0.0, float(c), cx_new, cy_new, w_new, h_new,
                self.snr_fill, self.psnr_fill
            ])

        img = img.to(torch.float32)
        targets = torch.tensor(targets_list, dtype=torch.float32) if targets_list else torch.zeros((0,8), dtype=torch.float32)
        return {"img": img, "targets": targets}

    @staticmethod
    def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        imgs = torch.stack([b["img"] for b in batch], dim=0)  # (B,C,T,T)
        all_tgts = []
        for i, b in enumerate(batch):
            t = b["targets"]
            if t.numel():
                t = t.clone()
                t[:,0] = i  # image index
                all_tgts.append(t)
        targets = torch.cat(all_tgts, dim=0) if all_tgts else torch.zeros((0,8), dtype=torch.float32)
        return imgs, targets, []

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
import random 
import math 
from typing import List, Optional, Tuple, Union

def plot_batch_with_boxes(
    feats,              # List of [B,1,H,W] tensors
    targets,            # List of predicted boxes [cls, x1, y1, x2, y2]
    class_names=None,
    max_batch_size=3,
    cmap='viridis',
    save_path=None,
    optimal_shape=(1024, 1024),
    labels=None         # ← GT boxes [cls, xc, yc, w, h] normalized
):
    H_opt, W_opt = optimal_shape
    C = len(feats)
    B = min(len(targets), max_batch_size)

    for i in range(B):
        pred_boxes = targets[i].tolist()

        chans = []
        for img in feats:
            if img.dim() == 4 and img.size(1) == 1:
                arr = img[i].squeeze(0).cpu().numpy()
            elif img.dim() == 3:
                arr = img[i].cpu().numpy()
            else:
                raise ValueError(f"Attendu (B,1,H,W) ou (B,H,W), got {tuple(img.shape)}")
            chans.append((arr, *arr.shape))

        fig, axes = plt.subplots(C, 1, figsize=(6, 4*C))
        if C == 1:
            axes = [axes]

        for c, (arr, Hc, Wc) in enumerate(chans):
            sx, sy = Wc / W_opt, Hc / H_opt
            ax = axes[c]
            ax.imshow(arr, aspect='auto', cmap=cmap)
            ax.set_title(f"Sample {i} • Résolution {c}: {Hc}×{Wc}")
            ax.axis('off')

            # === Dessiner ground truth (vert) ===
            if labels is not None and i < len(labels):
                for obj in labels[i]["labels"]:
                    xc, yc, w, h = obj["xc"], obj["yc"], obj["w"], obj["h"]
                    cls = obj["class"]
                    x1 = (xc - w / 2) * Wc
                    y1 = (yc - h / 2) * Hc
                    ax.add_patch(patches.Rectangle((x1, y1), w * Wc, h * Hc,
                                                   linewidth=1, edgecolor='lime', facecolor='none'))
                    label = class_names[int(cls)] if class_names else str(int(cls))
                    ax.text(x1, y1 + 5, label, color='black', fontsize=9,
                            bbox=dict(facecolor='lime', alpha=0.6, pad=1))
                    
            # === Dessiner prédictions (rouge) ===
            for cls, x1o, y1o, x2o, y2o in pred_boxes:
                x1 = x1o * sx; y1 = y1o * sy
                w = (x2o - x1o) * sx
                h = (y2o - y1o) * sy
                ax.add_patch(patches.Rectangle((x1, y1), w, h,
                                               linewidth=1, edgecolor='r', facecolor='none'))
                label = class_names[int(cls)] if class_names else str(int(cls))
                ax.text(x1, y1 - 5, label, color='white', fontsize=9,
                        bbox=dict(facecolor='red', alpha=0.6, pad=1))

        plt.tight_layout()

        if isinstance(save_path, str):
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            out = save_path.replace('.png', f'_sample{i}.png')
            fig.savefig(out)
            plt.close(fig)
            print(f"Saved plot to {out}")
        else:
            plt.show()
            plt.close(fig)

        

def plot_batch_matched_boxes(
    imgs,                    # per‐sample multi‐résolution : List[List[Tensor(Hc,Wc)]]
    gt_boxes_list,           # List[Tensor(N,4)] en ABS pixels relatifs à optimal_shape
    pred_boxes_list,         # List[Tensor(M,4)] idem
    anchors_list=None,       # List[Tensor(M,2)] idem, ou None
    optimal_shape=(1024, 1024),
    cmap='viridis',
    save_path="matched.png",
    max_batch_size=1
):
    """
    Affiche chaque résolution sur une ligne, et repasse les labels
    (GT, préd, ancres) de l'espace 1024×1024 vers (Hc,Wc).
    """

    print(len(imgs), imgs[1])
    H_opt, W_opt = optimal_shape
    B = min(len(imgs), max_batch_size)

    for i in range(B):
        chans = imgs[i]           # liste de C tensors (Hc,Wc) ou (1,Hc,Wc)
        gt    = gt_boxes_list[i]  # Tensor (N,4): [x1,y1,x2,y2] en pixels 0→1024
        pr    = pred_boxes_list[i]
        ancs  = anchors_list[i] if anchors_list is not None else None

        C = len(chans)
        fig, axes = plt.subplots(C, 1, figsize=(10, 4*C))
        if C == 1:
            axes = [axes]

        for c, chan in enumerate(chans):
            ax = axes[c]
            # --- dimensions du canal ---
            if chan.ndim == 3:
                # (1,Hc,Wc) ou (C,Hc,Wc)
                Hc, Wc = chan.shape[-2:]
            elif chan.ndim == 2:
                Hc, Wc = chan.shape
            else:
                raise ValueError(f"chan.ndim={chan.ndim}")

            sx, sy = Wc / W_opt, Hc / H_opt

            # --- préparation image 2D ---
            if chan.ndim == 3 and chan.shape[0] == 1:
                arr = chan.squeeze(0).cpu().numpy()
            elif chan.ndim == 3:
                arr = chan.mean(dim=0).cpu().numpy()
            else:
                arr = chan.cpu().numpy()
            ax.imshow(arr, aspect='auto', cmap=cmap)
            ax.set_title(f"Res {c} : {Hc}×{Wc}")
            ax.axis('off')

            # --- boîtes GT (rouge) ---
            for x1o, y1o, x2o, y2o in gt.tolist():
                x1 = x1o * sx
                y1 = y1o * sy
                w  = (x2o - x1o) * sx
                h  = (y2o - y1o) * sy
                ax.add_patch(patches.Rectangle(
                    (x1, y1), w, h,
                    edgecolor='red', facecolor='none', lw=1
                ))

            # --- boîtes préd (lime, --) + ancres (croix) ---
            for j, (x1o, y1o, x2o, y2o) in enumerate(pr.tolist()):
                x1 = x1o * sx
                y1 = y1o * sy
                w  = (x2o - x1o) * sx
                h  = (y2o - y1o) * sy
                ax.add_patch(patches.Rectangle(
                    (x1, y1), w, h,
                    edgecolor='lime', facecolor='none',
                    linestyle='--', lw=1
                ))
                if ancs is not None and j < len(ancs):
                    cx_o, cy_o = ancs[j].tolist()
                    cx, cy = cx_o * sx, cy_o * sy
                    ax.scatter(cx, cy, s=20, marker='+', edgecolor='lime')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out_path = save_path.replace(".png", f"_sample{i}.png")
        fig.savefig(out_path)
        plt.close(fig)


def plot_predicted_boxes_batch(imgs, batch_pred_boxes, save_path="predictions_batch.png", color='blue', cmap='Greys', max_boxes=100, max_batch_size=3):
    """
    Affiche et sauvegarde un batch d'images avec leurs boîtes prédictives.

    Args:
        imgs (Tensor): (B, C, H, W) - images du batch.
        batch_pred_boxes (List[Tensor]): Liste de Tensors (N_i, 4) - boîtes [x1, y1, x2, y2] pour chaque image.
        save_path (str): Chemin de sauvegarde.
        show_indices (bool): Si True, affiche les indices des boîtes.
        color (str): Couleur des boîtes (par défaut: 'blue').
        cmap (str): Colormap pour l'affichage.
        max_boxes (int): Nombre max de boîtes affichées par image.
    """
    assert imgs.ndim == 4, f"Expected image batch with shape (B, C, H, W), got {imgs.shape}"
    
    B = min(imgs.shape[0], max_batch_size)
    rows = math.ceil(math.sqrt(B))
    cols = math.ceil(B / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if B > 1 else [axes]

    for i in range(B):
        ax = axes[i]
        img_np = imgs[i].squeeze(0).cpu().numpy()
        ax.imshow(img_np, cmap=cmap)
        ax.axis('off')

        pred_boxes = batch_pred_boxes[i]
        total_boxes = len(pred_boxes)

        selected_indices = random.sample(range(total_boxes), max_boxes) if total_boxes > max_boxes else list(range(total_boxes))

        for j, idx in enumerate(selected_indices):
            x1, y1, x2, y2 = pred_boxes[idx][:4].tolist()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none',
                                     label='Prediction' if j == 0 else "")
            ax.add_patch(rect)

    for j in range(B, len(axes)):
        fig.delaxes(axes[j])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
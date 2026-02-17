from .metrics import match_boxes_iou, compute_ap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import torch
from tqdm import tqdm
from matplotlib import rcParams
import json
from typing import Optional, List, Dict, Any, Tuple, Sequence, Union
from thop import profile

# ─────────────────────────────────────────────────────────────
# 1) Analyse d’une image  ➜  TP / FP / FN avec score, label, snr
# ─────────────────────────────────────────────────────────────
def analyse_results(pred_boxes:  torch.Tensor,
                    pred_scores: torch.Tensor,
                    pred_labels: torch.Tensor,
                    gt_boxes:    torch.Tensor,
                    gt_labels:   torch.Tensor,
                    gt_snrs:     torch.Tensor,
                    iou_thresh:  float = 0.1,
                    gt_ids: Optional[torch.Tensor] = None,
                    gt_psnrs: Optional[Union[torch.Tensor, list, tuple]] = None,
                    psnr_keys: Optional[List[str]] = ["cfg128","cfg256","cfg512","cfg1024","cfg2048"]):
    """
    Retourne dict {tp:[…], fp:[…], fn:[…]} avec 'score'.
    Ajoute: 'snr', 'psnr' (si dispo) et les dimensions GT 'gt_wh'=[w,h] (en px).
    - gt_psnrs: (N, R) éventuel
    - psnr_keys: si fourni et de longueur R, on retourne un dict {key: val}; sinon une liste.
    """

    matches: List[Tuple[int,int,float]] = match_boxes_iou(pred_boxes, gt_boxes, iou_thresh)
    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}

    def _maybe_id(i):
        return int(gt_ids[i]) if gt_ids is not None else None

    def _psnr_payload(i: int):
        if gt_psnrs is None or gt_psnrs.numel() == 0:
            return []

        row = gt_psnrs[i]  
        row = row.detach().cpu()

        if row.ndim == 0:
            val = float(row.item())
            if psnr_keys is not None and len(psnr_keys) == 1:
                return {psnr_keys[0]: val}
            return [val]

        # Cas 1-D : vecteur (R,)
        r = row.view(-1)
        if psnr_keys is not None and len(psnr_keys) == r.numel():
            return {k: float(v) for k, v in zip(psnr_keys, r.tolist())}
        else:
            return [float(v) for v in r.tolist()]

    def _gt_wh(i: int):
        # gt_boxes: [x1,y1,x2,y2] → [w,h]
        b = gt_boxes[i]
        return float(b[2] - b[0]), float(b[3] - b[1])

    # ---------- TP ----------
    tps: List[Dict[str, Any]] = []
    for p_idx, g_idx, max_iou in matches:
        w, h = _gt_wh(g_idx)
        rec: Dict[str, Any] = dict(
            pred_box=pred_boxes[p_idx].tolist(),
            score=float(pred_scores[p_idx]),
            gt_box=gt_boxes[g_idx].tolist(),
            gt_wh=[w, h],
            label=int(pred_labels[p_idx]),
            gt_label=int(gt_labels[g_idx]),
            snr=float(gt_snrs[g_idx]),
            max_iou=max_iou
        )
        psnr_payload = _psnr_payload(g_idx)
        if psnr_payload != []:
            rec["psnr"] = psnr_payload
        gid = _maybe_id(g_idx)
        if gid is not None:
            rec["gt_idx"] = gid
        tps.append(rec)

    # ---------- FP ----------
    fps: List[Dict[str, Any]] = []
    for i in range(len(pred_boxes)):
        if i not in matched_p:
            fps.append(dict(
                pred_box=pred_boxes[i].tolist(),
                score=float(pred_scores[i]),
                label=int(pred_labels[i])
            ))

    # ---------- FN ----------
    fns: List[Dict[str, Any]] = []
    for i in range(len(gt_boxes)):
        if i not in matched_g:
            w, h = _gt_wh(i)
            rec: Dict[str, Any] = dict(
                gt_box=gt_boxes[i].tolist(),
                gt_wh=[w, h],
                label=int(gt_labels[i]),
                snr=float(gt_snrs[i])
            )
            psnr_payload = _psnr_payload(i)
            if psnr_payload != []:
                rec["psnr"] = psnr_payload
            gid = _maybe_id(i)
            if gid is not None:
                rec["gt_idx"] = gid
            fns.append(rec)

    return dict(tp=tps, fp=fps, fn=fns)


def analyse_dataset(model,
                    val_loader,
                    conf_thresh: float = 0.1,
                    iou_thresh : float = 0.1, 
                    img_size : int = 1024,
                    psnr_keys: Optional[List[str]] = ["cfg128","cfg256","cfg512","cfg1024","cfg2048"]):
    """
    Agrège TP / FP / FN sur val_loader et retourne un seul dictionnaire.
    Les TP / FP contiennent le 'score' de confiance.
    Ajoute le PSNR des GT si présent dans targets (colonnes après snr).
    - psnr_keys: optionnel, ex. ["cfg128","cfg256","cfg512","cfg1024","cfg2048"]
                 pour nommer les colonnes PSNR dans les sorties.
    """
    device = model.device
    agg: Dict[str, List[Dict[str, Any]]] = dict(tp=[], fp=[], fn=[])

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyse Dataset"):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, targets, batch_res_keys = batch
                if psnr_keys is None and batch_res_keys is not None:
                    psnr_keys = list(batch_res_keys)
            else:
                imgs, targets = batch

            imgs    = imgs.to(device) if isinstance(imgs, torch.Tensor) else [i.to(device) for i in imgs]
            targets = targets.to(device)

            d_out, c_out = model(imgs)
            preds = model.postprocess(d_out, c_out, d_out, conf_thresh)

            for img_i, det in enumerate(preds):
                t = targets[targets[:, 0] == img_i]
                if len(t):
                    xc, yc, w, h = t[:,2], t[:,3], t[:,4], t[:,5]
                    x1 = (xc - w/2) 
                    y1 = (yc - h/2) 
                    x2 = (xc + w/2) 
                    y2 = (yc + h/2) 
                    gt_boxes  = torch.stack([x1, y1, x2, y2], 1).cpu()
                    gt_labels = t[:,1].long().cpu()
                    gt_snrs   = t[:,6].cpu() if t.shape[1] > 6 else torch.zeros(len(t))

                    if t.shape[1] > 7:
                        gt_psnrs = t[:, 7:].cpu()  # (N, R)
                    else:
                        gt_psnrs = torch.zeros((len(t), 0)).cpu()
                else:
                    gt_boxes  = torch.zeros((0,4))
                    gt_labels = torch.zeros((0,), dtype=torch.long)
                    gt_snrs   = torch.zeros((0,))
                    gt_psnrs  = torch.zeros((0,0))

                if det.numel():
                    pred_boxes = (det[:, :4] / float(img_size)).cpu()
                    pred_scores = det[:, 4].cpu()          # scores
                    pred_labels = det[:, 5].long().cpu()
                else:
                    pred_boxes  = torch.zeros((0,4))
                    pred_scores = torch.zeros((0,))
                    pred_labels = torch.zeros((0,), dtype=torch.long)

                res = analyse_results(pred_boxes, pred_scores, pred_labels,
                                      gt_boxes, gt_labels, gt_snrs,
                                      iou_thresh,
                                      gt_psnrs=gt_psnrs,
                                      psnr_keys=psnr_keys)
                                                      
                agg["tp"].extend(res["tp"])
                agg["fp"].extend(res["fp"])
                agg["fn"].extend(res["fn"])

    return agg


def precision_recall_stats(
    stats,
    thresholds=np.linspace(0.0, 1.0, 50),
    to_plot=False,
    with_classes=False,
    class_index_to_name=None
):
    """
    Calcule recall, precision, f1 pour chaque seuil, globalement et par classe.

    Args:
        stats: dict with 'tp', 'fp', 'fn'
        thresholds: array of confidence thresholds
        to_plot: False | True | str (path to save plot)
        with_classes: bool, whether to plot per-class curves
        class_index_to_name: dict[int, str], optional mapping for class labels

    Returns:
        dict with recall, precision, f1 globally and per class.
    """
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]

    all_labels = sorted(set(d["label"] for d in tp + fp + fn))
    gt_total = len(tp) + len(fn)
    recalls, precisions, f1s = [], [], []

    gt_total_by_class = {cls: sum(1 for d in tp + fn if d["label"] == cls) for cls in all_labels}
    per_class = {
        cls: {"recall": [], "precision": [], "f1": []}
        for cls in all_labels
    }

    for thr in tqdm(thresholds, desc="P-R thresholds"):
        tp_thr = sum(1 for d in tp if d["score"] >= thr)
        fp_thr = sum(1 for d in fp if d["score"] >= thr)

        rec  = tp_thr / (gt_total + 1e-9)
        prec = tp_thr / (tp_thr + fp_thr + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9) if (prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)

        for cls in all_labels:
            tp_cls = sum(1 for d in tp if d["score"] >= thr and d["label"] == cls)
            fp_cls = sum(1 for d in fp if d["score"] >= thr and d["label"] == cls)
            gt_cls = gt_total_by_class[cls]
            rec_c = tp_cls / (gt_cls + 1e-9)
            prec_c = tp_cls / (tp_cls + fp_cls + 1e-9)
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9) if (prec_c + rec_c) > 0 else 0.0

            per_class[cls]["recall"].append(rec_c)
            per_class[cls]["precision"].append(prec_c)
            per_class[cls]["f1"].append(f1_c)

    # Extrémités
    recalls[0], precisions[0], f1s[0] = 1, 0, 0
    recalls[-1], precisions[-1], f1s[-1] = 0, 1, 0
    for cls in all_labels:
        per_class[cls]["recall"][0] = 1
        per_class[cls]["precision"][0] = 0
        per_class[cls]["f1"][0] = 0
        per_class[cls]["recall"][-1] = 0
        per_class[cls]["precision"][-1] = 1
        per_class[cls]["f1"][-1] = 0

    # ──────────────────────── PLOT ────────────────────────
    if to_plot:
        rcParams.update({
            'font.family': 'serif',
            'font.size': 13,
            'axes.labelsize': 14,
            'axes.titlesize': 15,
            'legend.fontsize': 11,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        cmap = plt.get_cmap("tab20", len(all_labels))

        def class_name(cls):
            return class_index_to_name[cls] if class_index_to_name and cls in class_index_to_name else str(cls)

        # 1. Precision vs Recall
        axs[0].plot(recalls, precisions, label="Global", color='black', linewidth=2)
        if with_classes:
            for i, cls in enumerate(all_labels):
                axs[0].plot(per_class[cls]["recall"], per_class[cls]["precision"],
                            label=f"{class_name(cls)}", linestyle=':', alpha=0.7, color=cmap(i))
        axs[0].set_title("Precision vs Recall")
        axs[0].set_xlabel("Recall")
        axs[0].set_ylabel("Precision")
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend(loc="lower left", ncol=2)

        # 2. Recall & Precision vs threshold
        axs[1].plot(thresholds, recalls, label="Recall", color='blue', linewidth=2)
        axs[1].plot(thresholds, precisions, label="Precision", color='red', linewidth=2)
        if with_classes:
            for i, cls in enumerate(all_labels):
                axs[1].plot(thresholds, per_class[cls]["recall"], linestyle='--', color=cmap(i), alpha=0.6, label=f"R-{class_name(cls)}")
                axs[1].plot(thresholds, per_class[cls]["precision"], linestyle=':', color=cmap(i), alpha=0.6, label=f"P-{class_name(cls)}")
        axs[1].set_title("Recall / Precision vs Confidence Threshold")
        axs[1].set_xlabel("Threshold")
        axs[1].set_ylabel("Score")
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].legend(ncol=2, loc="center right", fontsize=9)

        # 3. F1-score vs threshold
        axs[2].plot(thresholds, f1s, label="F1-score", color="green", linewidth=2)
        if with_classes:
            for i, cls in enumerate(all_labels):
                axs[2].plot(thresholds, per_class[cls]["f1"], linestyle=':', color=cmap(i), alpha=0.6, label=f"F1-{class_name(cls)}")
        axs[2].set_title("F1-score vs Confidence Threshold")
        axs[2].set_xlabel("Threshold")
        axs[2].set_ylabel("F1-score")
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].legend(ncol=2, loc="center right", fontsize=9)

        plt.tight_layout()
        if isinstance(to_plot, str):
            plt.savefig(to_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[✓] Figure enregistrée dans : {to_plot}")
        else:
            plt.show()

    return {
        "thr": list(thresholds),
        "recall": recalls,
        "precision": precisions,
        "f1": f1s,
        "per_class": {
            cls: {
                "recall": per_class[cls]["recall"],
                "precision": per_class[cls]["precision"],
                "f1": per_class[cls]["f1"]
            }
            for cls in per_class
        }
    }



def recall_per_snr_bin(
    stats,
    snr_bins,
    conf_thresh=0.2,
    to_plot=False,
    with_classes=False,
    class_index_to_name=None
):
    """
    Calcule le recall par bin de SNR, par classe et global. Affiche/sauvegarde optionnellement les courbes.

    Args:
        stats: dict contenant 'tp', 'fn'
        snr_bins: list of float, par ex [0, 2, 4, 6, 8, 10]
        conf_thresh: float, score minimum pour considérer un TP
        to_plot: False | True | str (chemin de sauvegarde du plot)
        with_classes: bool, pour afficher les courbes par classe
        class_index_to_name: dict[int, str], noms lisibles des classes (optionnel)

    Returns:
        {
            "per_class": {class_id: {"recall": np.array(n_bins), "snr_bins": snr_bins}},
            "global": {"recall": np.array(n_bins), "snr_bins": snr_bins}
        }
    """
    tp = stats["tp"]
    fn = stats["fn"]
    all_labels = sorted(set(d["label"] for d in tp + fn))
    snr_bins = np.array(snr_bins)
    n_bin = len(snr_bins) - 1
    snr_bin_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])

    result = {
        "per_class": {
            cls: {"recall": np.zeros(n_bin), "snr_bins": snr_bins}
            for cls in all_labels
        },
        "global": {"recall": np.zeros(n_bin), "snr_bins": snr_bins}
    }

    tps_arr = np.array(
        [(d["score"], d["label"], d["snr"]) for d in tp if d["score"] >= conf_thresh],
        dtype=[('score', 'f4'), ('label', 'i4'), ('snr', 'f4')]
    )
    fns_arr = np.array(
        [(d["label"], d["snr"]) for d in fn],
        dtype=[('label', 'i4'), ('snr', 'f4')]
    )

    for cls in all_labels:
        for bin_i in range(n_bin):
            snr_min, snr_max = snr_bins[bin_i], snr_bins[bin_i+1]
            tp_bin = tps_arr[(tps_arr["label"] == cls) & (snr_min <= tps_arr["snr"]) & (tps_arr["snr"] < snr_max)]
            fn_bin = fns_arr[(fns_arr["label"] == cls) & (snr_min <= fns_arr["snr"]) & (fns_arr["snr"] < snr_max)]
            denom = len(tp_bin) + len(fn_bin)
            recall = len(tp_bin) / (denom + 1e-9) if denom > 0 else np.nan
            result["per_class"][cls]["recall"][bin_i] = recall

    for bin_i in range(n_bin):
        snr_min, snr_max = snr_bins[bin_i], snr_bins[bin_i+1]
        tp_bin = tps_arr[(snr_min <= tps_arr["snr"]) & (tps_arr["snr"] < snr_max)]
        fn_bin = fns_arr[(snr_min <= fns_arr["snr"]) & (fns_arr["snr"] < snr_max)]
        denom = len(tp_bin) + len(fn_bin)
        recall = len(tp_bin) / (denom + 1e-9) if denom > 0 else np.nan
        result["global"]["recall"][bin_i] = recall

    # ──────────────── PLOT ────────────────
    if to_plot:
        rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(snr_bin_centers, result["global"]["recall"], label="Global", linewidth=2.5, color='black')

        if with_classes:
            cmap = plt.get_cmap("tab20", len(all_labels))
            for i, cls in enumerate(all_labels):
                recall_vals = result["per_class"][cls]["recall"]
                label_str = class_index_to_name[cls] if class_index_to_name and cls in class_index_to_name else f"Classe {cls}"
                ax.plot(snr_bin_centers, recall_vals, linestyle='--', color=cmap(i), label=label_str, alpha=0.75)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Recall")
        ax.set_title(f"Recall par bin de SNR (confidence ≥ {conf_thresh})")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', frameon=False)
        fig.tight_layout()

        if isinstance(to_plot, str):
            plt.savefig(to_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[✓] Figure enregistrée : {to_plot}")
        else:
            plt.show()

    return result


def map_from_stats(stats,
                   iou_thresholds=np.array([0.5, 0.75]),
                   map_iou_thresholds=np.arange(0.5, 1.0, 0.05),
                   to_plot=False,
                   class_index_to_name=None,
                   conf_thresh: float = 0.2,
                   with_classes: bool = True):
    """
    Calcule mAP50, mAP50:95 et tableau recall vs IoU pour chaque classe.
    Filtre les prédictions avec score >= conf_thresh.
    Affiche toujours le recall moyen, et les classes en pointillé si `with_classes=True`.
    """
    tp = [d for d in stats["tp"] if d["score"] >= conf_thresh]
    fp = [d for d in stats["fp"] if d["score"] >= conf_thresh]
    fn = stats["fn"]
    all_labels = sorted(set(d["label"] for d in tp + fp + fn))

    ap_all = {iou_thr: [] for iou_thr in map_iou_thresholds}
    recall_table = {
        cls: {iou_thr: 0.0 for iou_thr in iou_thresholds}
        for cls in all_labels
    }
    recall_per_iou = {iou_thr: [] for iou_thr in iou_thresholds}

    for cls in all_labels:
        pred = [(d["score"], d.get("max_iou", 0.0)) for d in tp + fp if d["label"] == cls]
        scores = np.array([s for s, _ in pred])
        ious = np.array([iou for _, iou in pred])

        n_gt = sum(1 for d in tp + fn if d["label"] == cls)
        if n_gt == 0:
            for iou_thr in map_iou_thresholds:
                ap_all[iou_thr].append(0.0)
            for iou_thr in iou_thresholds:
                recall_table[cls][iou_thr] = np.nan
            continue

        sort_idx = np.argsort(-scores)
        ious = ious[sort_idx]

        for iou_thr in map_iou_thresholds:
            tp_flags = ious >= iou_thr
            fp_flags = ~tp_flags
            tp_cumsum = np.cumsum(tp_flags)
            fp_cumsum = np.cumsum(fp_flags)
            recall = tp_cumsum / (n_gt + 1e-9)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
            prec_interp = np.maximum.accumulate(precision[::-1])[::-1]
            ap = np.trapz(prec_interp, recall)
            ap_all[iou_thr].append(ap)

        for iou_thr in iou_thresholds:
            tp_count = np.sum(ious >= iou_thr)
            recall_value = tp_count / (n_gt + 1e-9)
            recall_table[cls][iou_thr] = recall_value
            recall_per_iou[iou_thr].append(recall_value)

    mAP50 = np.mean(ap_all[0.5])
    mAP5095 = np.mean([np.mean(ap_all[t]) for t in map_iou_thresholds])

    mean_recall_per_iou = {
        iou_thr: np.nanmean(recall_per_iou[iou_thr]) for iou_thr in iou_thresholds
    }

    # ────── PLOT ──────
    if to_plot:
        print(f"[mAP50]     = {mAP50:.4f}")
        print(f"[mAP50:95]  = {mAP5095:.4f}")

        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 11
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")

        # Courbe moyenne (toujours affichée)
        iou_vals = list(mean_recall_per_iou.keys())
        recall_vals = list(mean_recall_per_iou.values())
        ax.plot(iou_vals, recall_vals, marker='o', linewidth=2, label="Recall moyen", color="black")

        if with_classes:
            for idx, cls in enumerate(all_labels):
                iou_vals_cls = list(recall_table[cls].keys())
                recall_vals_cls = list(recall_table[cls].values())
                label = class_index_to_name[cls] if class_index_to_name else f"Classe {cls}"
                ax.plot(iou_vals_cls, recall_vals_cls, alpha=0.6,
                        linestyle=':', label=label, color=cmap(idx % 10))

        ax.set_title("Recall vs Seuil IoU", pad=10)
        ax.set_xlabel("Seuil IoU")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc="lower left", frameon=True)
        ax.set_xticks(np.round(iou_thresholds, 2))

        fig.tight_layout()

        if isinstance(to_plot, str):
            out_path = Path(to_plot)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300)
        else:
            plt.show()

    return {
        "mAP50": mAP50,
        "mAP50:95": mAP5095,
        "recall_per_iou": recall_table,
        "mean_recall_per_iou": mean_recall_per_iou
    }


# ─────────────────────────────────────────────────────────────
# Matrice de confusion filtrée SNR
# ─────────────────────────────────────────────────────────────
def confusion_matrix_snr(stats,
                         snr_min: float,
                         snr_max: float,
                         class_index_to_name: dict,
                         to_plot=False,
                         conf_thresh: float = 0.2,
                         normalize: Optional[str] = None):  # 'row', 'col', or None
    """
    Matrice de confusion GT × Préd, filtrée par SNR et seuil de confiance.
    Dernière ligne et colonne = bruit.

    Args:
        stats: dict contenant 'tp', 'fp', 'fn'
        snr_min, snr_max: bornes SNR
        class_index_to_name: ex: {0: 'LFM', ..., 14: 'T4'}
        to_plot: False, True ou str (chemin)
        conf_thresh: seuil de confiance sur TP / FP
        normalize: None, 'row' ou 'col' pour normalisation ligne ou colonne

    Returns:
        np.ndarray shape [num_classes+1, num_classes+1]
    """
    noise_class = len(class_index_to_name)
    num_classes = noise_class + 1
    M = np.zeros((num_classes, num_classes), dtype=int)

    # ----------- TP -----------
    for d in tqdm(stats["tp"], desc="Confusion TP"):
        if snr_min <= d["snr"] <= snr_max and d.get("score", 1.0) >= conf_thresh:
            gt_label = d["gt_label"]
            pred_label = d.get("label", gt_label)
            M[gt_label, pred_label] += 1

    # ----------- FP -----------
    for d in tqdm(stats["fp"], desc="Confusion FP"):
        if d.get("score", 1.0) >= conf_thresh:
            snr_ok = "snr" not in d or snr_min <= d["snr"] <= snr_max
            if snr_ok:
                pred = d["label"]
                M[noise_class, pred] += 1

    # ----------- FN -----------
    for d in tqdm(stats["fn"], desc="Confusion FN"):
        if snr_min <= d["snr"] <= snr_max:
            gt = d["label"]
            M[gt, noise_class] += 1

    # ----------- NORMALISATION -----------
    if normalize == "row":
        M_float = M.astype(np.float32)
        row_sums = M_float.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        M_plot = M_float / row_sums
    elif normalize == "col":
        M_float = M.astype(np.float32)
        col_sums = M_float.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        M_plot = M_float / col_sums
    else:
        M_plot = M

    # ----------- PLOT -----------
    if to_plot:
        class_names = [class_index_to_name[i] for i in range(noise_class)] + ['bruit']
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(M_plot, annot=False, fmt="d", cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar=True,
                    square=True, linewidths=0.5, linecolor='gray',
                    ax=ax)

        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
        title = f"Confusion matrix (SNR ∈ [{snr_min}, {snr_max}], score ≥ {conf_thresh})"
        if normalize:
            title += f" — normalisée par {normalize}"
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        fig.tight_layout()
        if isinstance(to_plot, str):
            Path(to_plot).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(to_plot, dpi=300)
            print(f"[✓] Enregistré dans : {to_plot}")
        else:
            plt.show()

    return M


from collections import defaultdict
from typing import Dict, Any, List, Optional

def _extract_max_psnr(psnr_payload) -> Optional[float]:
    """Supporte psnr en dict({'cfg128':..}) ou list([...]). Retourne max ou None si absent."""
    if psnr_payload is None:
        return None
    if isinstance(psnr_payload, dict):
        vals = list(psnr_payload.values())
    elif isinstance(psnr_payload, list):
        vals = psnr_payload
    else:
        return None
    if len(vals) == 0:
        return None
    # Filtre les sentinelles (-1.0) si présentes
    vals = [v for v in vals if v is not None and np.isfinite(v) and v > -1.0]
    return max(vals) if len(vals) else None

def recall_per_size_bin(stats: Dict[str, Any],
                        size_bins_rel: Sequence[float] = tuple(np.linspace(0.0, 1.0, 101)),
                        conf_thresh: float = 0.0,
                        with_classes: bool = False,
                        class_index_to_name: Optional[Dict[int,str]] = None,
                        snr_min: Optional[float] = None) -> Dict[str, Any]:
    """
    Recall par taille RELATIVE avec binning sur la petite dimension min(w_rel,h_rel) ∈ [0,1].
    - stats: dict {tp:[...], fp:[...], fn:[...]} avec 'gt_wh_rel' et 'score' (TP), 'snr' (GT).
    - size_bins_rel: bornes de bins (croissantes) dans [0,1]; bords gauches inclusifs.
    - conf_thresh: filtre TP par score ≥ conf_thresh.
    - with_classes: si True, ajoute 'per_class'.
    - snr_min: si fourni, ne compte que les GT dont snr ≥ snr_min.
    """
    bins = np.asarray(size_bins_rel, dtype=float)
    assert np.all(np.diff(bins) >= 0.0), "size_bins_rel doit être trié croissant."
    assert bins[0] >= 0.0 and bins[-1] <= 1.0, "Les bornes doivent être dans [0,1]."

    K = len(bins) - 1
    tp_counts = np.zeros(K, dtype=int)
    fn_counts = np.zeros(K, dtype=int)

    per_class_tp = defaultdict(lambda: np.zeros(K, dtype=int)) if with_classes else None
    per_class_fn = defaultdict(lambda: np.zeros(K, dtype=int)) if with_classes else None

    # ---------- TP ----------
    for rec in stats.get("tp", []):
        if rec.get("score", 0.0) < conf_thresh:
            continue
        gt_snr = rec.get("snr", None)
        if snr_min is not None and (gt_snr is None or float(gt_snr) < snr_min):
            continue

        w_rel, h_rel = rec.get("gt_wh", [None, None])
        if w_rel is None or h_rel is None:
            continue
        size = float(min(w_rel, h_rel))  # relatif
        bin_idx = np.digitize(size, bins[1:], right=True)  # 0..K-1
        if 0 <= bin_idx < K:
            tp_counts[bin_idx] += 1
            if with_classes:
                cls_idx = int(rec.get("gt_label", rec.get("label", -1)))
                per_class_tp[cls_idx][bin_idx] += 1

    # ---------- FN ----------
    for rec in stats.get("fn", []):
        gt_snr = rec.get("snr", None)
        if snr_min is not None and (gt_snr is None or float(gt_snr) < snr_min):
            continue

        w_rel, h_rel = rec.get("gt_wh", [None, None])
        if w_rel is None or h_rel is None:
            continue
        size = float(min(w_rel, h_rel))
        bin_idx = np.digitize(size, bins[1:], right=True)
        if 0 <= bin_idx < K:
            fn_counts[bin_idx] += 1
            if with_classes:
                cls_idx = int(rec.get("gt_label", rec.get("label", -1)))
                per_class_fn[cls_idx][bin_idx] += 1

    denom = tp_counts + fn_counts
    recall = np.divide(tp_counts, np.maximum(denom, 1), dtype=float)

    out = {
        "bins_rel": bins.tolist(),
        "tp": tp_counts.tolist(),
        "fn": fn_counts.tolist(),
        "recall": recall.tolist(),
    }

    if with_classes:
        per_class = {}
        for cls, arr_tp in per_class_tp.items():
            arr_fn = per_class_fn[cls]
            denom_c = arr_tp + arr_fn
            rec_c = np.divide(arr_tp, np.maximum(denom_c, 1), dtype=float)
            name = class_index_to_name.get(cls, str(cls)) if class_index_to_name else str(cls)
            per_class[name] = {
                "tp": arr_tp.tolist(),
                "fn": arr_fn.tolist(),
                "recall": rec_c.tolist()
            }
        out["per_class"] = per_class

    return out

def recall_per_max_psnr_bin(stats: Dict[str, Any],
                            psnr_bins: List[float] = tuple(range(0, 31)),  # 0..30 dB par défaut
                            conf_thresh: float = 0.0,
                            with_classes: bool = False,
                            class_index_to_name: Optional[Dict[int,str]] = None) -> Dict[str, Any]:
    """
    Recall par max_psnr (dB) côté GT (TP: score ≥ conf_thresh).
    - psnr_bins: bords (entiers par défaut), k = len(bins)-1.
    - Ignorer les GT sans PSNR (non comptés ni en TP ni en FN).
    """
    K = len(psnr_bins) - 1
    tp_counts = np.zeros(K, dtype=int)
    fn_counts = np.zeros(K, dtype=int)
    per_class_tp = defaultdict(lambda: np.zeros(K, dtype=int)) if with_classes else None
    per_class_fn = defaultdict(lambda: np.zeros(K, dtype=int)) if with_classes else None

    # TP
    for rec in stats.get("tp", []):
        if rec.get("score", 0.0) < conf_thresh:
            continue
        maxp = _extract_max_psnr(rec.get("psnr"))
        if maxp is None:
            continue
        bin_idx = np.digitize(maxp, psnr_bins[1:], right=True)
        if 0 <= bin_idx < K:
            tp_counts[bin_idx] += 1
            if with_classes:
                per_class_tp[int(rec.get("gt_label", rec.get("label", -1)))][bin_idx] += 1

    # FN
    for rec in stats.get("fn", []):
        maxp = _extract_max_psnr(rec.get("psnr"))
        if maxp is None:
            continue
        bin_idx = np.digitize(maxp, psnr_bins[1:], right=True)
        if 0 <= bin_idx < K:
            fn_counts[bin_idx] += 1
            if with_classes:
                per_class_fn[int(rec.get("label", -1))][bin_idx] += 1

    denom = tp_counts + fn_counts
    recall = np.divide(tp_counts, np.maximum(denom, 1), dtype=float)

    out = {
        "bins": list(psnr_bins),
        "tp": tp_counts.tolist(),
        "fn": fn_counts.tolist(),
        "recall": recall.tolist(),
    }

    if with_classes:
        per_class = {}
        for cls, arr_tp in per_class_tp.items():
            arr_fn = per_class_fn[cls]
            denom_c = arr_tp + arr_fn
            rec_c = np.divide(arr_tp, np.maximum(denom_c, 1), dtype=float)
            name = class_index_to_name.get(cls, str(cls)) if class_index_to_name else str(cls)
            per_class[name] = {
                "tp": arr_tp.tolist(),
                "fn": arr_fn.tolist(),
                "recall": rec_c.tolist()
            }
        out["per_class"] = per_class

    return out


def dataset_analysis_with_metrics(model,
                                   val_loader,
                                   iou_thresh: float = 0.5,
                                   fa: float = 0.01,
                                   img_size: int = 1024,
                                   to_save=False,
                                   to_plot=False, 
                                   stats_path: Optional[Union[str, Path]] = None):

    CLASS_INDEX_TO_NAME = {
        0: 'no_mod', 1: 'LFM', 2: 'NLFM', 3: 'frank', 4: 'P1', 5: 'P2',
        6: 'P3', 7: 'P4', 8: 'random_biphasique', 9: 'FSK', 10: 'DSSS',
        11: 'T1', 12: 'T2', 13: 'T3', 14: 'T4'
    }

    # ─────── 0. Chargement / calcul des stats ───────
    stats = None
    if stats_path is not None:
        stats_path = Path(stats_path)
        if stats_path.exists():
            print(f"[INFO] Chargement des stats depuis {stats_path}")
            with open(stats_path, "r") as f:
                stats = json.load(f)

    if stats is None:
        print(f"[INFO] Calcul des stats avec analyse_dataset ...")
        stats = analyse_dataset(
            model,
            val_loader,
            conf_thresh=0.05,
            iou_thresh=iou_thresh, 
            img_size=img_size,
        )
        if stats_path is not None:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"[✓] Stats sauvegardées dans {stats_path.resolve()}")

    full_metrics = {}

    # ─────── 1. Courbes P/R + détection seuil optimal ───────
    F1_stats = precision_recall_stats(
        stats, 
        to_plot=to_plot, 
        with_classes=True, 
        class_index_to_name=CLASS_INDEX_TO_NAME
    )
    full_metrics['f1_stats'] = F1_stats

    # Choisir conf_thresh ≈ seuil où FP rate ≤ fa
    conf_thresh = 0.0
    for thr, prec in zip(F1_stats["thr"], F1_stats["precision"]):
        if (1 - prec) <= fa:
            conf_thresh = thr
            break

    print(f"[INFO] Seuil de confiance sélectionné pour FP ≤ {fa:.3f} : {conf_thresh:.3f}")

    # ─────── 2. Rappel par SNR ───────
    recall_snr = recall_per_snr_bin(
        stats,
        snr_bins=range(-15, 20),
        to_plot=to_plot,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        conf_thresh=conf_thresh
    )
    full_metrics['recall_snr'] = recall_snr

    # ─────── 2a. Recall par TAILLE (relatif) ───────
    recall_size = recall_per_size_bin(
        stats,
        conf_thresh=conf_thresh,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME, 
        snr_min=10
    )
    full_metrics['recall_size_minsnr10db'] = recall_size

    # ─────── 2b. Recall par MAX_PSNR (dB) ───────
    recall_max_psnr = recall_per_max_psnr_bin(
        stats,
        psnr_bins=list(range(0, 31)),  
        conf_thresh=conf_thresh,
        with_classes=True,
        class_index_to_name=CLASS_INDEX_TO_NAME
    )
    full_metrics['recall_max_psnr'] = recall_max_psnr

    # ─────── 3. mAP ───────
    map_stats = map_from_stats(
        stats,
        iou_thresholds=np.linspace(0, 1, 21),
        to_plot=to_plot,
        class_index_to_name=CLASS_INDEX_TO_NAME,
        conf_thresh=conf_thresh
    )
    full_metrics['map_stats'] = map_stats

    # ─────── 4. Matrices de confusion (normalisées) ───────
    for label, (snr_min, snr_max) in {
        "high": (10, 20),
        "medium": (0, 10),
        "low": (-10, 0)
    }.items():
        mat = confusion_matrix_snr(
            stats,
            snr_min=snr_min,
            snr_max=snr_max,
            class_index_to_name=CLASS_INDEX_TO_NAME,
            to_plot=to_plot,
            conf_thresh=conf_thresh,
            normalize='row'  
        )
        full_metrics[f"conf_matrix_{label}_snr"] = mat

    dummy_inputs = [
        torch.randn(1, 1, H, W).to(model.device)
        for (H, W) in model.input_resolutions
    ]

    # Calcul des FLOPs et paramètres
    try:
        macs, params = profile(model, inputs=(dummy_inputs,), verbose=False)
        full_metrics["model_info"] = {
            "params": int(params),
            "flops": int(macs)
        }
        print(f"[✓] FLOPs: {macs:,}  |  Params: {params:,}")
    except Exception as e:
        print(f"[⚠️] Impossible de calculer FLOPs/params : {e}")
        full_metrics["model_info"] = {
            "params": None,
            "flops": None,
            "error": str(e)
        }

    # ─────── 5. Sauvegarde des métriques complètes ───────
    if to_save:
        save_path = Path(to_save)

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(full_metrics, f, indent=2, default=convert)

        print(f"[✓] full_metrics sauvegardé en JSON dans : {save_path.resolve()}")

    return full_metrics

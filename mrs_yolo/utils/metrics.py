"""Model validation metrics."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import precision_score, recall_score, average_precision_score

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def match_boxes_iou(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Retourne une liste de tuples (pred_idx, gt_idx, iou) pour chaque match.
    Chaque prédiction ne peut matcher qu’un seul GT (et vice versa).
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    # Matrice des IoU : shape [N_pred, N_gt]
    ious = box_iou(pred_boxes, gt_boxes)  
    matched_gt = set()
    matches = []
    

    for pred_idx in range(ious.shape[0]):
        # trouver le GT de meilleure IoU pour cette préd
        row = ious[pred_idx]
        max_iou, gt_idx = row.max(dim=0)
        gt_idx = gt_idx.item()
        max_iou = max_iou.item()

        # valider le match si seuil et pas déjà pris
        if max_iou >= iou_thresh and gt_idx not in matched_gt:
            matched_gt.add(gt_idx)
            matches.append((pred_idx, gt_idx, max_iou))

    return matches



class ConfusionMatrix:
    def __init__(self, nc, iou_thres=0.5):
        self.nc = nc
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))

    def process(self, detections, labels):
        if len(labels) == 0:
            for *_, cls in detections:
                self.matrix[int(cls), self.nc] += 1
            return

        if len(detections) == 0:
            for *_, cls in labels:
                self.matrix[self.nc, int(cls)] += 1
            return

        det_boxes = torch.tensor(detections)[:, :4]
        det_cls = torch.tensor(detections)[:, 4].int()
        gt_boxes = torch.tensor(labels)[:, :4]
        gt_cls = torch.tensor(labels)[:, 4].int()

        ious = box_iou(gt_boxes, det_boxes)
        x = torch.where(ious > self.iou_thres)

        matched_gt, matched_det = [], []
        for gt_idx, det_idx in zip(*x):
            if gt_idx in matched_gt or det_idx in matched_det:
                continue
            self.matrix[det_cls[det_idx], gt_cls[gt_idx]] += 1
            matched_gt.append(gt_idx)
            matched_det.append(det_idx)

        for det_idx in range(len(det_cls)):
            if det_idx not in matched_det:
                self.matrix[det_cls[det_idx], self.nc] += 1

        for gt_idx in range(len(gt_cls)):
            if gt_idx not in matched_gt:
                self.matrix[self.nc, gt_cls[gt_idx]] += 1

    def print(self):
        print("Confusion Matrix:")
        print(self.matrix.astype(int))


def compute_confidence_vs_pfa(model, noise_loader, thresholds=np.linspace(0.01, 1.0, 100)):
    pfa_per_thresh = []

    model.eval()
    with torch.no_grad():
        all_confidences = []

        for imgs, _ in noise_loader:
            imgs = imgs.to(model.device)
            dist_out, clsobj_out = model(imgs)
            feats = dist_out
            outputs = model.postprocess(dist_out, clsobj_out, feats)

            for preds in outputs:
                if preds is not None and len(preds) > 0:
                    confs = preds[:, 4]  # assume 5th col is confidence
                    all_confidences.extend(confs.cpu().numpy())

        all_confidences = np.array(all_confidences)
        total_preds = len(all_confidences)

        for thresh in thresholds:
            pfa = np.sum(all_confidences >= thresh) / (total_preds + 1e-6)
            pfa_per_thresh.append(pfa)

    return thresholds, pfa_per_thresh


def plot_conf_vs_pfa(thresholds, pfas, save_path):
    plt.figure()
    plt.plot(thresholds, pfas)
    plt.xlabel("Confidence threshold")
    plt.ylabel("PFA (False Alarm Probability)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def compute_ap(all_dets, all_gts, iou_thres: float = 0.5) -> float:
    """
    Average Precision (AP) à un seuil d'IoU donné, sur l'ensemble des classes.
    - all_dets: list de tuples (boxes, scores, labels) par image
    - all_gts:  list de tuples (boxes,        labels) par image
    """
    # Regrouper par classe
    class_data = {}
    for img_idx, ((db, ds, dl), (gb, gl)) in enumerate(zip(all_dets, all_gts)):
        for box, score, cls in zip(db, ds, dl):
            class_data.setdefault(int(cls), {"scores": [], "boxes": [], "img_idx": [], "gt_boxes": {}})
            class_data[int(cls)]["scores"].append(score)
            class_data[int(cls)]["boxes"].append(box)
            class_data[int(cls)]["img_idx"].append(img_idx)
        for box, cls in zip(gb, gl):
            cd = class_data.setdefault(int(cls), {"scores": [], "boxes": [], "img_idx": [], "gt_boxes": {}})
            cd["gt_boxes"].setdefault(img_idx, []).append(box)

    ap_per_class = []
    for cls, data in class_data.items():
        scores = np.array(data["scores"])
        order = scores.argsort()[::-1]
        boxes = [data["boxes"][i] for i in order]
        img_ids = [data["img_idx"][i] for i in order]
        gt_boxes = data["gt_boxes"]

        detected = {img: np.zeros(len(bxs)) for img, bxs in gt_boxes.items()}
        tp = np.zeros(len(boxes))
        fp = np.zeros(len(boxes))

        for i, (box, img_id) in enumerate(zip(boxes, img_ids)):
            gts = gt_boxes.get(img_id, [])
            if gts:
                # calcul IoU avec tous les GT de l'image
                ious = [float(x) for x in box_iou(torch.tensor(box[None]), torch.tensor(np.stack(gts)))[0]]
                best_iou = max(ious)
                best_j  = int(np.argmax(ious))
                if best_iou >= iou_thres and detected[img_id][best_j] == 0:
                    tp[i] = 1
                    detected[img_id][best_j] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # courbe PR
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        n_gt    = sum(len(v) for v in gt_boxes.values())
        if n_gt == 0:
            continue
        recall    = tp_cum / (n_gt + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        # interpolation 11 points
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p
        ap_per_class.append(ap / 11.0)

    return float(np.mean(ap_per_class)) if ap_per_class else 0.0


def compute_ar(all_dets, all_gts, K: int = 100, iou_thres: float = 0.5) -> float:
    """
    Average Recall @K :
    - on prend top-K détections par image, on compte TP / nombre de GT, puis moyenne sur images.
    """
    recalls = []
    for (db, ds, dl), (gb, gl) in zip(all_dets, all_gts):
        n_gt = len(gb)
        if n_gt == 0:
            continue
        # top-K
        order = np.argsort(ds)[::-1][:K]
        dets = [db[i] for i in order]

        matched = np.zeros(n_gt)
        tp = 0
        for det in dets:
            ious = [float(x) for x in box_iou(torch.tensor(det[None]), torch.tensor(gb))[0]]
            if ious and max(ious) >= iou_thres:
                j = int(np.argmax(ious))
                if matched[j] == 0:
                    tp += 1
                    matched[j] = 1

        recalls.append(tp / (n_gt + 1e-6))

    return float(np.mean(recalls)) if recalls else 0.0
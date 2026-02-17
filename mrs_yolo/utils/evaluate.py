# yolo_perso/runners/eval_runner.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
import os

from yolo_perso.utils.analysing_results import dataset_analysis_with_metrics

@dataclass
class EvalConfig:
    iou_thresh: float = 0.5
    fa_target: float = 0.01
    img_size: int = 1024

class EvalRunner:
    """
    - Fournit les noms des colonnes additionnelles (extra headers)
    - Exécute l’analyse dataset périodiquement
    - Résume les métriques en scalaires alignés avec les extra headers
    - Sauvegarde un JSON complet par epoch
    """
    def __init__(self, output_dir: str, cfg: EvalConfig):
        self.output_dir = Path(output_dir)
        self.cfg = cfg

    # --- Colonnes additionnelles (stables, indépendantes du run effectif) ---
    def extra_headers(self) -> List[str]:
        return [
            "map50",
            "map50_95",
            "avg_recall_low_snr",     # [-10, 0]
            "avg_recall_medium_snr",  # [0, 10]
            "avg_recall_high_snr",    # [10, s_max]
            "metrics_json_path",
        ]

    @staticmethod
    def _avg_recall_between(snr_bins: np.ndarray, recall: np.ndarray, a: float, b: float) -> float:
        """
        Moyenne de recall sur [a, b] en supposant la valeur constante par bin.
        Intègre la partie du bin chevauchant [a, b].
        """
        snr_bins = np.asarray(snr_bins, dtype=float)
        recall = np.asarray(recall, dtype=float)

        s_min, s_max = snr_bins[0], snr_bins[-1]
        left = max(a, s_min)
        right = min(b, s_max)
        if right <= left:
            return float("nan")

        area = 0.0
        for k in range(len(recall)):
            L, R = snr_bins[k], snr_bins[k + 1]
            ov_l = max(L, left)
            ov_r = min(R, right)
            width = max(0.0, ov_r - ov_l)
            if width > 0:
                area += recall[k] * width

        denom = max(right - left, 1e-12)
        return float(area / denom)

    # --- Résumé -> scalaires alignés avec extra_headers() ---
    def _summarize(self, full_metrics: Dict[str, Any]) -> Dict[str, Any]:
        # -- mAP
        map_stats = full_metrics.get("map_stats", {})
        map50 = map_stats.get("mAP50")
        map50_95 = map_stats.get("mAP50:95")

        snr_bins = full_metrics['recall_snr']['global']['snr_bins']
        recall_curve = full_metrics['recall_snr']['global']['recall']

        avg_low = avg_med = avg_high = float("nan")
        if snr_bins is not None and recall_curve is not None:
            snr_bins = np.asarray(snr_bins, dtype=float)
            recall_curve = np.asarray(recall_curve, dtype=float)

            s_max = float(snr_bins[-1])
            avg_low  = self._avg_recall_between(snr_bins, recall_curve, -10.0,  s_max)
            avg_med  = self._avg_recall_between(snr_bins, recall_curve,   0.0, s_max)
            avg_high = self._avg_recall_between(snr_bins, recall_curve, 10.0, s_max)

        return {
            "map50": map50,
            "map50_95": map50_95,
            "avg_recall_low_snr": avg_low,
            "avg_recall_medium_snr": avg_med,
            "avg_recall_high_snr": avg_high,
        }

    def _save_full_metrics_json(self, full_metrics: Dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj
        with open(path, "w") as f:
            json.dump(full_metrics, f, indent=2, default=convert)
        return path

    def run(self, epoch: int, model, val_loader) -> Dict[str, Any]:
        extra_headers = self.extra_headers()

        full_metrics = dataset_analysis_with_metrics(
            model=model,
            val_loader=val_loader,
            iou_thresh=self.cfg.iou_thresh,
            fa=self.cfg.fa_target,
            img_size=self.cfg.img_size,
            to_save=False,
            to_plot=False,
        )

        summary = self._summarize(full_metrics)

        json_path = self.output_dir / "metrics" /f"metrics_epoch_{epoch:03d}.json"
        self._save_full_metrics_json(full_metrics, json_path)

        extra_values = [
            summary["map50"],
            summary["map50_95"],
            summary["avg_recall_low_snr"],
            summary["avg_recall_medium_snr"],
            summary["avg_recall_high_snr"],
            str(json_path),
        ]

        return {
            "did_eval": True,
            "extra_headers": extra_headers,
            "extra_values": extra_values,
            "json_path": str(json_path),
            "full_metrics": full_metrics,
        }


class MetricsLogger:
    """
    - Crée automatiquement le CSV si absent
    - Écrit toujours les colonnes de base
    - Les colonnes additionnelles viennent d’EvalRunner (noms + valeurs)
    """
    BASE_HEADERS = [
        "epoch",
        "train_loss", "val_loss",
        "loss_box_train", "loss_cls_train", "loss_dfl_train",
        "loss_box_val",   "loss_cls_val",   "loss_dfl_val",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._header_written = os.path.exists(csv_path)
        self._current_header: Optional[List[str]] = None

    def _ensure_header(self, extra_headers: Optional[List[str]] = None):
        extra_headers = extra_headers or []
        full_header = self.BASE_HEADERS + extra_headers

        if not self._header_written:
            # Création du fichier + header
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(full_header)
            self._header_written = True
            self._current_header = full_header
        else:
            # Si le fichier existe déjà, on vérifie (optionnel) la compatibilité
            if self._current_header is None:
                with open(self.csv_path, "r", newline="") as f:
                    r = csv.reader(f)
                    first_row = next(r, None)
                self._current_header = first_row or full_header
            # Si des colonnes supplémentaires sont demandées après coup, on ne modifie pas le header existant.
            if self._current_header != full_header:
                pass

    def log(
        self,
        epoch: int,
        train_loss: float, val_loss: float,
        loss_box_train: float, loss_cls_train: float, loss_dfl_train: float,
        loss_box_val: float,   loss_cls_val: float,   loss_dfl_val: float,
        extra_headers: Optional[List[str]] = None,
        extra_values: Optional[List[Any]] = None,
    ):
        self._ensure_header(extra_headers)

        row_common = [
            epoch,
            train_loss, val_loss,
            loss_box_train, loss_cls_train, loss_dfl_train,
            loss_box_val,   loss_cls_val,   loss_dfl_val,
        ]

        extra_values = extra_values or []
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(row_common + extra_values)


class TrainingPlots:
    """
    Professional plots for papers:
      - Loss vs Epochs (train & val)
      - mAP50 & mAP50_95 vs Epochs
      - Average recall per SNR band vs Epochs (low/med/high)
    Design notes:
      - single-column figure size by default (IEEE-ish): ~3.4" x 2.4"
      - colorblind-safe palette (Okabe–Ito)
      - grid with minor ticks, subtle spines, readable fonts
      - vector export (PDF) alongside raster (PNG)
    """

    # Okabe–Ito palette (colorblind-safe)
    _C = {
        "black":   "#000000",
        "orange":  "#E69F00",
        "sky":     "#56B4E9",
        "green":   "#009E73",
        "yellow":  "#F0E442",
        "blue":    "#0072B2",
        "verm":    "#D55E00",
        "purple":  "#CC79A7",
        "grey":    "#6F6F6F",
    }

    @staticmethod
    def _export(fig: plt.Figure, save_path: str, dpi: int = 300):
        """Save PNG only."""
        save_path = str(save_path)
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        
    @staticmethod
    def _apply_ax_style(ax: plt.Axes, xlabel: str, ylabel: str, title: Optional[str] = None):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # grid & ticks
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # spines & tick params
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_linewidth(0.8)
            ax.spines[s].set_alpha(0.9)
        ax.tick_params(axis="both", which="both", labelsize=8, length=3)
        ax.tick_params(axis="both", which="major", length=4.5)

        # legend (compat toutes versions)
        handles, labels = ax.get_legend_handles_labels()
        if handles and any(labels):
            leg = ax.legend(handles, labels, frameon=False, handlelength=2.2, handletextpad=0.6)
            # régler l’alpha sur les handles directement (pas via leg.legendHandles)
            for h in handles:
                try:
                    h.set_alpha(0.95)
                except Exception:
                    pass


    @staticmethod
    def _read_column(csv_path: str, col_name: str) -> Tuple[List[float], List[float]]:
        epochs, values = [], []
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r)
            if not header or col_name not in header:
                return epochs, values
            i_epoch = header.index("epoch")
            i_col = header.index(col_name)
            for row in r:
                try:
                    e = float(row[i_epoch])
                    v_str = row[i_col].strip()
                    if v_str == "" or v_str.lower() == "none":
                        continue
                    v = float(v_str)
                except Exception:
                    continue
                epochs.append(e)
                values.append(v)
        return epochs, values

    # --- Styling context -----------------------------------------------------
    @staticmethod
    def paper_style(use_tex: bool = False,
                    base_fontsize: int = 9,
                    width_in: float = 3.4,
                    height_in: float = 2.4):
        """
        Context manager to apply a consistent paper style.
        use_tex=True requires a LaTeX install. Default is False for portability.
        """
        class _Ctx:
            def __enter__(self):
                self._old = mpl.rcParams.copy()
                mpl.rcParams.update({
                    "figure.figsize": (width_in, height_in),
                    "savefig.dpi": 300,
                    "font.size": base_fontsize,
                    "axes.titlesize": base_fontsize,
                    "axes.labelsize": base_fontsize,
                    "xtick.labelsize": base_fontsize - 1,
                    "ytick.labelsize": base_fontsize - 1,
                    "legend.fontsize": base_fontsize - 1,
                    "axes.formatter.use_mathtext": True,
                    "text.usetex": use_tex,
                    "font.family": "serif" if use_tex else "sans-serif",
                    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
                    "lines.linewidth": 2.0,
                    "axes.prop_cycle": mpl.cycler(color=[
                        TrainingPlots._C["blue"],
                        TrainingPlots._C["orange"],
                        TrainingPlots._C["green"],
                        TrainingPlots._C["purple"],
                        TrainingPlots._C["sky"],
                        TrainingPlots._C["verm"],
                        TrainingPlots._C["grey"],
                    ]),
                })
                return self
            def __exit__(self, exc_type, exc, tb):
                mpl.rcParams.update(self._old)
        return _Ctx()

    # --- Public plotting APIs ------------------------------------------------
    @staticmethod
    def plot_losses(csv_path: str, save_path: str, use_tex: bool = False):
        e_tr, v_tr = TrainingPlots._read_column(csv_path, "train_loss")
        e_va, v_va = TrainingPlots._read_column(csv_path, "val_loss")

        with TrainingPlots.paper_style(use_tex=use_tex):
            fig, ax = plt.subplots()
            # Markers every ~10 points for readability
            def _me(vals): return max(1, len(vals) // 10)
            if e_tr and v_tr:
                ax.plot(e_tr, v_tr, label="Train loss",
                        marker="o", markersize=3.2, markevery=_me(v_tr))
            if e_va and v_va:
                ax.plot(e_va, v_va, label="Val loss",
                        linestyle="--", marker="s", markersize=3.0, markevery=_me(v_va))
            TrainingPlots._apply_ax_style(ax, xlabel="Epoch", ylabel="Loss", title="Loss vs Epochs")

            # Optionally annotate the best val loss
            if v_va:
                i_best = int(np.argmin(v_va))
                ax.plot(e_va[i_best], v_va[i_best], marker="*", markersize=6.5,
                        color=TrainingPlots._C["verm"], label="Best val")
                ax.legend(frameon=False)

            TrainingPlots._export(fig, save_path)
            plt.close(fig)

    @staticmethod
    def plot_maps(csv_path: str, save_path: str, use_tex: bool = False):
        e_m50, v_m50 = TrainingPlots._read_column(csv_path, "map50")
        e_m95, v_m95 = TrainingPlots._read_column(csv_path, "map50_95")

        with TrainingPlots.paper_style(use_tex=use_tex):
            fig, ax = plt.subplots()
            def _me(vals): return max(1, len(vals) // 10)
            if e_m50 and v_m50:
                ax.plot(e_m50, v_m50, label="mAP@50",
                        marker="o", markersize=3.2, markevery=_me(v_m50))
            if e_m95 and v_m95:
                ax.plot(e_m95, v_m95, label="mAP@50:95",
                        linestyle="--", marker="s", markersize=3.0, markevery=_me(v_m95))
            TrainingPlots._apply_ax_style(ax, xlabel="Epoch", ylabel="mAP", title="mAP vs Epochs")

            # Annotate best mAP50:95 if available
            if v_m95:
                i_best = int(np.nanargmax(v_m95))
                ax.plot(e_m95[i_best], v_m95[i_best], marker="*", markersize=6.5,
                        color=TrainingPlots._C["verm"], label="Best mAP@50:95")
                ax.legend(frameon=False)

            TrainingPlots._export(fig, save_path)
            plt.close(fig)

    @staticmethod
    def plot_avg_recalls(csv_path: str, save_path: str, use_tex: bool = False):
        e_low,  v_low  = TrainingPlots._read_column(csv_path, "avg_recall_low_snr")
        e_med,  v_med  = TrainingPlots._read_column(csv_path, "avg_recall_medium_snr")
        e_high, v_high = TrainingPlots._read_column(csv_path, "avg_recall_high_snr")

        with TrainingPlots.paper_style(use_tex=use_tex):
            fig, ax = plt.subplots()
            def _me(vals): return max(1, len(vals) // 10)
            if e_low and v_low:
                ax.plot(e_low, v_low,  label="Avg recall (low SNR)",
                        marker="o", markersize=3.2, markevery=_me(v_low))
            if e_med and v_med:
                ax.plot(e_med, v_med,  label="Avg recall (medium SNR)",
                        linestyle="--", marker="s", markersize=3.0, markevery=_me(v_med))
            if e_high and v_high:
                ax.plot(e_high, v_high, label="Avg recall (high SNR)",
                        linestyle="-.", marker="^", markersize=3.2, markevery=_me(v_high))

            TrainingPlots._apply_ax_style(ax, xlabel="Epoch", ylabel="Average recall",
                                          title="Average Recall per SNR band")
            TrainingPlots._export(fig, save_path)
            plt.close(fig)

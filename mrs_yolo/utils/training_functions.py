import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import numpy as np
import json 
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from collections import defaultdict


def should_stop_early_from_csv(
    log_path: str,
    patience: int = 5,
    monitor: str = "val_loss",
    mode: str = "min",
    min_delta: float = 0.0,
) -> bool:
    """
    Early stopping basé sur un CSV de logs.
    - mode='min'  : on s'arrête si la métrique n'a pas DIMINUÉ d'au moins min_delta
    - mode='max'  : on s'arrête si la métrique n'a pas AUGMENTÉ d'au moins min_delta
    Les lignes vides/None/Nan sont ignorées.
    """
    try:
        with open(log_path, "r") as f:
            rows = list(csv.DictReader(f))
        # extraire les valeurs valides
        vals = []
        for row in rows:
            v = row.get(monitor, "")
            try:
                v = float(v)
                if np.isnan(v):
                    continue
                vals.append(v)
            except Exception:
                continue

        if len(vals) < patience + 1:
            return False  # pas assez d'epochs valides

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        if mode == "min":
            best = float("inf")
            no_improve = 0
            for v in vals:
                if v < best - min_delta:
                    best = v
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        return True
        else:  # mode == "max"
            best = float("-inf")
            no_improve = 0
            for v in vals:
                if v > best + min_delta:
                    best = v
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        return True

        return False
    except Exception as e:
        print(f"[EarlyStopping] Erreur lors de la lecture du CSV : {e}")
        return False


def plot_training_curves_from_csv(log_path, save_path=None):
    """
    Trace 4 sous-graphiques montrant l'évolution des différentes composantes de la loss
    (loss totale, loss_box, loss_cls, loss_dfl) pour l'entraînement et la validation.

    Args:
        log_path (str): Chemin du fichier CSV contenant les colonnes.
        save_path (str, optional): Chemin où sauvegarder le graphique.
    """
    epochs = []
    train_losses = []
    val_losses = []
    box_train = []
    box_val = []
    cls_train = []
    cls_val = []
    dfl_train = []
    dfl_val = []

    try:
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["train_loss"] and row["val_loss"]:
                    epochs.append(int(row["epoch"]))
                    train_losses.append(float(row["train_loss"]))
                    val_losses.append(float(row["val_loss"]))
                    box_train.append(float(row.get("loss_box_train", 0)))
                    box_val.append(float(row.get("loss_box_val", 0)))
                    cls_train.append(float(row.get("loss_cls_train", 0)))
                    cls_val.append(float(row.get("loss_cls_val", 0)))
                    dfl_train.append(float(row.get("loss_dfl_train", 0)))
                    dfl_val.append(float(row.get("loss_dfl_val", 0)))
    except Exception as e:
        print(f"[Plot] Erreur lors de la lecture du CSV : {e}")
        return

    if not epochs:
        print("[Plot] Aucun log valide trouvé.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # Plot 1 - Total Loss
    axs[0].plot(epochs, train_losses, label="Train Loss", linewidth=2)
    axs[0].plot(epochs, val_losses, label="Val Loss", linewidth=2)
    axs[0].set_title("Total Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2 - Box Loss
    axs[1].plot(epochs, box_train, label="Box Train", linewidth=2)
    axs[1].plot(epochs, box_val, label="Box Val", linewidth=2)
    axs[1].set_title("Box Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3 - Class Loss
    axs[2].plot(epochs, cls_train, label="Cls Train", linewidth=2)
    axs[2].plot(epochs, cls_val, label="Cls Val", linewidth=2)
    axs[2].set_title("Classification Loss")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Loss")
    axs[2].grid(True)
    axs[2].legend()

    # Plot 4 - DFL Loss
    axs[3].plot(epochs, dfl_train, label="DFL Train", linewidth=2)
    axs[3].plot(epochs, dfl_val, label="DFL Val", linewidth=2)
    axs[3].set_title("DFL Loss (Distance Focal Loss)")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Loss")
    axs[3].grid(True)
    axs[3].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Plot] Courbes sauvegardées dans {save_path}")
    else:
        plt.show()


def plot_metrics_from_csv(csv_path, save_dir=None, snr_bin_size=5):
    df = pd.read_csv(csv_path)
    epoch = df["epoch"]

    # 📊 Plot des métriques scalaires
    scalar_metrics = ["recall", "precision", "f1_score", "accuracy", "balanced_accuracy", "map50", "pd"]
    plt.figure(figsize=(10, 6))
    for metric in scalar_metrics:
        if metric in df.columns:
            plt.plot(epoch, df[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Scalar Metrics over Epochs")
    plt.grid(True)
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "scalar_metrics.png"))
    else:
        plt.show()
    plt.close()

    # 📈 Traitement de la colonne pd_by_snr
    if "pd_by_snr" in df.columns:
        pd_by_snr_all = []
        snr_keys_set = set()

        for i, s in enumerate(df["pd_by_snr"]):
            try:
                val = json.loads(s.replace("'", "\""))
                if isinstance(val, dict):
                    pd_by_snr_all.append(val)
                    snr_keys_set.update(val.keys())
                elif isinstance(val, list):
                    val = {str(i): v for i, v in enumerate(val)}
                    pd_by_snr_all.append(val)
                    snr_keys_set.update(val.keys())
                else:
                    raise ValueError("Unsupported type")
            except Exception as e:
                print(f"[⚠️ Warning] Failed to parse pd_by_snr row {i}: {e}")
                pd_by_snr_all.append({})

        if not snr_keys_set:
            print("[⚠️ Warning] No valid SNR keys found in 'pd_by_snr'. Skipping Pd by SNR plot.")
            return

        def group_snr_keys(snr_keys, bin_size=snr_bin_size):
            snr_ints = sorted([int(k) for k in snr_keys])
            min_snr, max_snr = min(snr_ints), max(snr_ints)
            bins = list(range(min_snr, max_snr + bin_size, bin_size))
            snr_groups = defaultdict(list)
            for k in snr_keys:
                snr = int(k)
                for i in range(len(bins) - 1):
                    if bins[i] <= snr < bins[i + 1]:
                        label = f"{bins[i]} to {bins[i + 1] - 1}"
                        snr_groups[label].append(k)
                        break
            return snr_groups

        snr_groups = group_snr_keys(snr_keys_set, snr_bin_size)

        plt.figure(figsize=(10, 6))
        for group_label, keys in snr_groups.items():
            group_values = []
            for epoch_vals in pd_by_snr_all:
                group_pd = np.nanmean([epoch_vals.get(k, np.nan) for k in keys])
                group_values.append(group_pd)
            plt.plot(epoch, group_values, label=group_label)

        plt.xlabel("Epoch")
        plt.ylabel("Pd")
        plt.title("Pd by SNR group over Epochs")
        plt.grid(True)
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "pd_by_snr.png"))
        else:
            plt.show()
        plt.close()

def plot_confusion_matrix(confmat, class_names=None, normalize=False, save_path=None, title="Confusion Matrix"):
    """
    Affiche ou sauvegarde une matrice de confusion.

    Args:
        confmat (np.ndarray or torch.Tensor): Matrice de confusion carrée (num_classes x num_classes).
        class_names (list of str): Noms des classes (facultatif).
        normalize (bool): Si True, normalise ligne par ligne.
        save_path (str): Chemin de sauvegarde (si None, affiche à l'écran).
        title (str): Titre de la figure.
    """
    if isinstance(confmat, torch.Tensor):
        confmat = confmat.cpu().numpy()

    if normalize:
        confmat = confmat.astype("float") / (confmat.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confmat, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else ".0f", ax=ax)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pd_vs_snr(pd_by_snr_json: str,
                   save_path: str = None,
                   title: str = "Probability of Detection vs SNR"):
    """
    Trace la probabilité de détection (Pd) en fonction du SNR.

    Args:
        pd_by_snr_json (str): JSON-encoded dict mapping SNR (as str or int) → Pd (float).
        save_path (str): chemin de sauvegarde de la figure (PNG). Si None, affiche à l’écran.
        title (str): titre de la figure.
    """
    # Parse the JSON into a dict
    try:
        pd_dict = json.loads(pd_by_snr_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for pd_by_snr: {e}")

    # Convert keys to floats and values to floats
    snrs = []
    pds  = []
    for k, v in pd_dict.items():
        try:
            snr = float(k)
            pd_val = float(v)
        except ValueError:
            continue
        snrs.append(snr)
        pds.append(pd_val)

    if not snrs:
        print("[⚠] Aucun point PD/SNR valide à tracer.")
        return

    # Sort by increasing SNR
    order = np.argsort(snrs)
    snrs = np.array(snrs)[order]
    pds  = np.array(pds)[order]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(snrs, pds, marker='o', linestyle='-')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Detection (Pd)")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()
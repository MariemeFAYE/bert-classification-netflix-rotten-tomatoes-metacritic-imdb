"""
utils.py

Fonctions utilitaires pour le projet :
- fixation de la seed (reproductibilité)
- calcul des métriques (accuracy, F1-score)
- visualisations (courbes d'apprentissage, matrice de confusion)
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def set_seed(seed: int = 42) -> None:
    """Fixe la seed pour random, numpy et torch afin de rendre les résultats reproductibles:
        seed: La valeur de la graine aléatoire (par défaut 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred) -> dict:
    """Calcule l'accuracy et le F1-score (macro) à partir des labels réels et prédits.

    Le F1 macro traite les deux classes à égalité, ce qui est adapté à notre
    dataset déséquilibré (~3:1) : il évalue honnêtement la classe minoritaire (Series).

        y_true: Labels réels (liste ou tableau d'entiers).
        y_pred: Labels prédits (liste ou tableau d'entiers).

    Retourne: Un dictionnaire contenant 'accuracy' et 'f1'.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_curves(history: dict, save_path: str = "curves.png") -> None:
    """Trace les courbes de loss et d'accuracy/F1 (train vs validation).

        history: Dictionnaire des métriques par epoch
                 (clés : train_loss, val_loss, train_acc, val_acc, val_f1).
        save_path: Chemin de sauvegarde de la figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sous-figure 1 : la loss (pour diagnostiquer l'overfitting)
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Courbe de perte (Loss)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Sous-figure 2 : accuracy et F1
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val accuracy")
    axes[1].plot(epochs, history["val_f1"], "^-", label="Val F1 (macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / F1-score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Courbes sauvegardées : {save_path}")


def plot_confusion(y_true, y_pred, class_names, save_path: str = "confusion_matrix.png") -> None:
    """Trace et sauvegarde la matrice de confusion.

        y_true: Labels réels.
        y_pred: Labels prédits.
        class_names: Liste des noms de classes (dans l'ordre des indices).
        save_path: Chemin de sauvegarde de la figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Matrice de confusion (validation)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Matrice de confusion sauvegardée : {save_path}")
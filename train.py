"""
train.py

Fine-tuning d'un modèle BERT pour la classification Series / Movie
à partir des résumés Netflix.

Boucle d'entraînement, Contient :
  - le chargement du CSV et le split train/validation 80/20 stratifié
  - les fonctions train_epoch / eval_epoch (loss, accuracy, F1)
  - la sauvegarde du meilleur modèle (best val_loss)
  - le suivi des métriques avec Weights & Biases (wandb)

Usage :
    python train.py
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from dataset import TextClassificationDataset
from model import BertClassifier
from utils import set_seed, compute_metrics, plot_curves, plot_confusion


def load_data(csv_path):
    """Charge le CSV et prépare textes, labels et le dictionnaire de labels :
        csv_path: Chemin du CSV brut Netflix.

    Retourne: Tuple (texts, labels, labels_dict, class_names).
    """
    df = pd.read_csv(csv_path)
    # On garde le résumé (texte) et la cible (Series or Movie)
    df = df[["Summary", "Series or Movie"]].copy()
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Dictionnaire label -> entier, construit une fois et partagé train/val
    class_names = sorted(df["label"].unique().tolist())  # ['Movie', 'Series']
    labels_dict = {name: idx for idx, name in enumerate(class_names)}

    return texts, labels, labels_dict, class_names


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    """Effectue une epoch d'entraînement :
        model: Le modèle BertClassifier.
        dataloader: DataLoader d'entraînement.
        optimizer: Optimiseur (AdamW).
        scheduler: Scheduler de learning rate.
        loss_fn: Fonction de perte (CrossEntropyLoss).
        device: 'cuda' ou 'cpu'.

    Retourne: Tuple (train_loss, train_accuracy).
    """
    model.train()  # mode entraînement (dropout actif)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()                 # remise à zéro des gradients
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)        # CrossEntropyLoss 
        loss.backward()                       # rétropropagation
        optimizer.step()                      # mise à jour des poids
        scheduler.step()                      # mise à jour du learning rate

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = compute_metrics(all_labels, all_preds)["accuracy"]
    return avg_loss, acc


def eval_epoch(model, dataloader, loss_fn, device):
    """Évalue le modèle sur l'ensemble de validation :

        model: Le modèle BertClassifier.
        dataloader: DataLoader de validation.
        loss_fn: Fonction de perte.
        device: 'cuda' ou 'cpu'.

    Retourne: Tuple (val_loss, val_accuracy, val_f1, y_true, y_pred).
    """
    model.eval()  # mode évaluation (dropout désactivé)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():  # pas de calcul de gradients en validation
        for batch in tqdm(dataloader, desc="Val", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1"], all_labels, all_preds


def main():
    # Point d'entrée : prépare les données, fine-tune BERT, sauvegarde le meilleur modèle.
    parser = argparse.ArgumentParser(description="Fine-tuning BERT (Series / Movie)")
    parser.add_argument("--csv", type=str, default="data/netflix-rotten-tomatoes-metacritic-imdb 3.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    args = parser.parse_args()

    # Reproductibilité
    set_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train] Device : {device}")

    # Suivi wandb
    wandb.init(project="bert-netflix-classification", config=vars(args))

    # Chargement des données
    texts, labels, labels_dict, class_names = load_data(args.csv)
    print(f"[train] Exemples : {len(texts)} | Classes : {labels_dict}")

    # Split train / validation 80/20 
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,  # conserve la proportion des classes
    )
    print(f"[train] Train : {len(train_texts)} | Validation : {len(val_texts)}")

    # Datasets et DataLoaders 
    train_ds = TextClassificationDataset(
        train_texts, train_labels, args.model_name, args.max_length, labels_dict
    )
    val_ds = TextClassificationDataset(
        val_texts, val_labels, args.model_name, args.max_length, labels_dict
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Modèle, loss, optimiseur, scheduler
    model = BertClassifier(args.model_name, n_class=len(labels_dict)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10 % de warmup
        num_training_steps=total_steps,
    )

    # Boucle d'entraînement 
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    y_true, y_pred = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_acc, val_f1, y_true, y_pred = eval_epoch(model, val_loader, loss_fn, device)
        current_lr = scheduler.get_last_lr()[0]

        # Affichage des métriques demandées à chaque epoch
        print(f"train_loss={train_loss:.4f} | train_accuracy={train_acc:.4f}")
        print(f"val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f} | val_f1_score={val_f1:.4f} | learning_rate={current_lr:.2e}")

        # Historique pour les courbes
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Log wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_f1_score": val_f1,
            "learning_rate": current_lr,
        })

        # Sauvegarde du meilleur modèle (best val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"[train] Meilleur modèle sauvegardé (val_loss={val_loss:.4f}) -> {args.save_path}")

    # Visualisations finales (pour le README) 
    plot_curves(history, save_path="curves.png")
    plot_confusion(y_true, y_pred, class_names, save_path="confusion_matrix.png")

    print(f"\n[train] Terminé. Meilleure val_loss : {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
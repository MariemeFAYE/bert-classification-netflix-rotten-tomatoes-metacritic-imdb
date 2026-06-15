"""
inspect_dataset.py

Analyse exploratoire du dataset avant le fine-tuning BERT.

Objectifs :
    - Nombre d'exemples
    - Nombre de classes
    - Distribution des classes
    - Longueur des textes
    - Exemples annotés
    - Histogramme des longueurs
"""

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


CSV_PATH = "data/netflix-rotten-tomatoes-metacritic-imdb 3.csv"
MODEL_NAME = "bert-base-uncased"


def inspect_dataset():

    print("=" * 80)
    print("CHARGEMENT DU DATASET")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)

    df = df[["Summary", "Series or Movie"]].copy()
    df.columns = ["text", "label"]

    df = df.dropna(subset=["text", "label"])

    df["text"] = df["text"].astype(str)

    print(f"Nombre total d'exemples : {len(df)}")

    print("\n")

    print("=" * 80)
    print("CLASSES")
    print("=" * 80)

    classes = sorted(df["label"].unique())

    print(f"Nombre de classes : {len(classes)}")
    print(classes)

    print("\n")

    print("=" * 80)
    print("DISTRIBUTION DES CLASSES")
    print("=" * 80)

    distribution = df["label"].value_counts()

    print(distribution)

    ratio = distribution.max() / distribution.min() 

    print(f"\nRatio déséquilibre : {ratio:.2f}")

    if ratio > 2:
        print("Dataset déséquilibré (>2:1)")
    else:
        print("Dataset relativement équilibré")

    print("\n")

    print("=" * 80)
    print("EXEMPLES")
    print("=" * 80)

    for i in range(min(5, len(df))):

        print(f"\nExemple {i+1}")
        print(f"Label : {df.iloc[i]['label']}")
        print(f"Texte : {df.iloc[i]['text'][:300]}")

    print("\n")

    print("=" * 80)
    print("ANALYSE DES LONGUEURS")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = []

    for text in df["text"]:

        tokens = tokenizer.encode(
            text,
            add_special_tokens=True
        )

        lengths.append(len(tokens))

    print(f"Longueur min : {min(lengths)} tokens")
    print(f"Longueur max : {max(lengths)} tokens")
    print(f"Longueur moyenne : {sum(lengths)/len(lengths):.2f} tokens")

    print("\nPercentiles :")

    print(f"50%  : {pd.Series(lengths).quantile(0.50):.0f}")
    print(f"90%  : {pd.Series(lengths).quantile(0.90):.0f}")
    print(f"95%  : {pd.Series(lengths).quantile(0.95):.0f}")
    print(f"99%  : {pd.Series(lengths).quantile(0.99):.0f}")

    print("\n")

    print("=" * 80)
    print("RECOMMANDATION MAX_LENGTH")
    print("=" * 80)

    p95 = int(pd.Series(lengths).quantile(0.95))

    if p95 <= 128:
        recommended = 128
    elif p95 <= 256:
        recommended = 256
    else:
        recommended = 512

    print(f"max_length recommandé : {recommended}")

    print("\n")

    print("=" * 80)
    print("SAUVEGARDE DES GRAPHIQUES")
    print("=" * 80)

    plt.figure(figsize=(8, 5))

    plt.hist(lengths, bins=30)

    plt.xlabel("Nombre de tokens")
    plt.ylabel("Fréquence")
    plt.title("Distribution des longueurs des textes")

    plt.tight_layout()

    plt.savefig(
        "token_length_distribution.png",
        dpi=150
    )

    plt.close()

    distribution.plot(
        kind="bar",
        figsize=(6, 4)
    )

    plt.title("Distribution des classes")

    plt.tight_layout()

    plt.savefig(
        "class_distribution.png",
        dpi=150
    )

    plt.close()

    print("✓ token_length_distribution.png généré")
    print("✓ class_distribution.png généré")


if __name__ == "__main__":
    inspect_dataset()
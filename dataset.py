""" 
Dataset PyTorch personnalisé pour la classification de texte avec BERT. 
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte.

    Chaque élément retourné contient :

        {
            "input_ids": Tensor[max_length],
            "attention_mask": Tensor[max_length],
            "label": Tensor[]
        }

    Compatible avec les modèles BERT de Hugging Face.
    """

    def __init__(self, texts, labels, model_name, max_length, labels_dict):
        """
        Cette méthode initialise le dataset.
        Arguments:
            texts : Liste des textes.

            labels :  Liste des labels sous forme texte 

            model_name  : Nom du tokenizer Hugging Face.

            max_length : Longueur maximale des séquences.

            labels_dict (dict): Mapping label -> entier.

                Exemple : {
                    "Movie": 0,
                    "Series": 1
                }
        """

        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.labels_dict = labels_dict

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        """
        Nombre total d'exemples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retourne un exemple tokenisé.

        Arguments:
            idx : Index de l'exemple.

        Retours:
            dict
        """

        text = str(self.texts[idx])

        label = self.labels_dict[self.labels[idx] ]

        encoding = self.tokenizer( text, add_special_tokens=True, max_length=self.max_length, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

    def get_class_distribution(self):
        """
        Retourne la distribution des classes.

        Retours:
            dict
        """

        distribution = {}

        for label in self.labels:
            distribution[label] = (
                distribution.get(label, 0) + 1
            )

        return distribution

    def compute_token_statistics(self):
        """
        Calcule les statistiques de longueur.

        Permet de justifier le choix de
        max_length dans le rapport.

        Returns:
            dict
        """

        lengths = []

        for text in self.texts:

            tokens = self.tokenizer.encode(
                str(text),
                add_special_tokens=True
            )

            lengths.append(len(tokens))

        return {
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "avg_tokens": sum(lengths) / len(lengths)
        }
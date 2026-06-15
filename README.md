# Fine-Tuning de BERT pour la Classification de Contenus Netflix

## Binôme

- Étudiant 1 : FAYE MARIEME
- Étudiant 2 : NDIAYE ABDOURAHMANE

Master Intelligence Artificielle – Deep Learning

---

# 1. Présentation du projet

L'objectif de ce projet est de fine-tuner un modèle BERT pré-entraîné afin de classifier automatiquement les contenus Netflix en deux catégories :

- Movie
- Series

Le projet est réalisé en PyTorch en implémentant manuellement la boucle d'entraînement, sans utiliser le Trainer de Hugging Face.

# 2. Dataset

## 2.1 Description

Dataset Netflix Rotten Tomatoes Metacritic IMDb.

Variable cible : **Series or Movie**

Variable explicative : **Summary**

## 2.2 Statistiques générales

| Indicateur | Valeur |
|------------|---------|
| Nombre total d'exemples | 15471 |
| Nombre de classes | 2 |
| Classe 1 | Movie | 
| Classe 2 | Series | 

## 2.3 Distribution des classes

| Classe | Nombre |
|----------|--------|
| Movie | 11689 |
| Series | 3782 |

<img src="class_distribution.png" width="600">


## 2.4 Analyse des longueurs des textes

| Mesure | Valeur |
|---------|---------|
| Longueur minimale | 64 tokens |
| Longueur maximale | 32 tokens |
| Longueur moyenne | 31.98  tokens |
| 95e percentile | 38 tokens |

Choix retenu : max_length = 128

<img src="token_length_distribution.png" width="600">

# 3. Architecture du modèle

Modèle utilisé : **bert-base-uncased**

Architecture :

Texte → Tokenizer → BERT → [CLS] → Couche Linéaire → Prédiction

# 4. Prétraitement des données

- Tokenisation avec AutoTokenizer
- Padding
- Truncation
- Attention Mask
- Encodage des labels

Movie → 0

Series → 1

# 5. Découpage Train / Validation

- 80 % entraînement
- 20 % validation

Split stratifié.

# 6. Hyperparamètres

| Paramètre | Valeur |
|------------|---------|
| Modèle | bert-base-uncased |
| Batch Size | 16 |
| Epochs | 4 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Optimiseur | AdamW |
| Max Length | 128 |

# 7. Méthodologie d'entraînement

Calcul des métriques :

- train_loss
- val_loss
- train_accuracy
- val_accuracy
- val_f1_score

Sauvegarde du meilleur modèle selon la validation loss.

# 8. Résultats
 
##  Train Accuracy 
    98,3 %
##  Train Loss 
    5,2 %
## Validation Accuracy 
    83,4 %
## F1-score
    76,8 $
## Validation loss 
    61 %

## Matrice de confusion

<img src="confusion_matrix.png" width="600">

## Courbes

<img src="curves.png" width="600">

# 9. Interface Gradio

Fonctionnalités :

- Saisie d'un résumé
- Prédiction Movie / Series
- Probabilités des classes
- Exemples pré-remplis

Lancement :

```bash
python demo.py
```

# 10. Difficultés rencontrées

- Compréhension de BERT
- Gestion de la tokenisation
- Optimisation de l'entraînement
- Déploiement avec Gradio

# 11. Répartition du travail

## Marieme FAYE

- Modèle BERT (model.py)
- Entraînement (train.py)
- Utilitaire (utils.py)
- Readme

## Abdourahmane NDIAYE

- Analyse exploratoire  (inspect_dataset.py)
- Dataset (dataset.py)
- Interface web avec gradio (demo.py)
- Readme

# 12. Installation

```bash
git clone https://github.com/MariemeFAYE/bert-classification-netflix-rotten-tomatoes-metacritic-imdb
cd bert-classification-netflix-rotten-tomatoes-metacritic-imdb
pip install -r requirements.txt
```

# 13. Exécution

## Analyse

```bash
python inspect_dataset.py
```

## Entraînement

```bash
python train.py
```

## Démo

```bash
python demo.py
```

# 14. Conclusion

Ce projet a permis de mettre en œuvre un pipeline complet de classification de texte basé sur BERT, depuis l'analyse des données jusqu'au déploiement via Gradio.

"""
model.py

Définition du modèle de classification basé sur BERT.

On charge un BERT pré-entraîné (transfer learning) et on ajoute une tête de
classification linéaire (nn.Linear) au-dessus du vecteur du premier token,
qui sert de représentation de toute la séquence.
"""

import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """Modèle BERT + tête linéaire pour la classification de texte.

    Le vecteur du premier token sert de représentation de tout le texte, puis une couche linéaire projette ce
    vecteur vers les scores (logits) des classes.
    """

    def __init__(self, model_name_or_path, n_class):
        """Initialise le modèle.
            model_name_or_path: Nom du modèle BERT pré-entraîné (ex. "bert-base-uncased").
            n_class: Nombre de classes (nous avons 2 : Movie / Series).
        """
        super().__init__()
        # BERT pré-entraîné (transfer learning) : ses poids seront fine-tunés.
        self.pretrained = BertModel.from_pretrained(model_name_or_path)
        # Dimension des vecteurs de sortie de BERT (768 pour bert-base).
        dim_in = self.pretrained.config.hidden_size
        # Tête de classification : projette le vecteur de la séquence vers n_class scores.
        self.proj_lin = nn.Linear(dim_in, n_class)

    def forward(self, input_ids, attention_mask):
        """Passe avant : du texte tokenizé vers les logits des classes.
            input_ids: Tenseur des identifiants de tokens, forme [batch, max_length].
            attention_mask: Masque d'attention (1 = token réel, 0 = padding),
                            forme [batch, max_length].

        Retourne: Les logits, tenseur de forme [batch, n_class].
        """
        
        # On passe les tokens et le masque d'attention à BERT.
        outputs = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state a la forme [batch, max_length, hidden_size].
        # On récupère le vecteur du premier token (position 0) comme représentation
        # de toute la séquence -> forme [batch, hidden_size].
        pooled = outputs.last_hidden_state[:, 0, :]
        # La tête linéaire projette ce vecteur vers les scores (logits) des classes.
        logits = self.proj_lin(pooled)
        return logits
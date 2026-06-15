"""
Interface Gradio pour la classification Movie / Series à partir des résumés Netflix.
 
"""

import torch
import torch.nn.functional as F
import gradio as gr

from transformers import AutoTokenizer

from model import BertClassifier


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "best_model.pt"

CLASS_NAMES = [
    "Movie",
    "Series"
]

#DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
     DEVICE = torch.device("cuda")
else:
     DEVICE = torch.device("cpu")
print(f"[train] Device : {DEVICE}")
# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ==========================================================
# MODEL
# ==========================================================

model = BertClassifier(
    MODEL_NAME,
    n_class=len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("[demo] Modèle chargé avec succès")


# ==========================================================
# PREDICTION
# ==========================================================

def predict(text):

    if text is None or len(text.strip()) == 0:
        return "Texte vide", {}

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )

    probabilities = probabilities.cpu().numpy()[0]

    predicted_idx = probabilities.argmax()

    predicted_class = CLASS_NAMES[
        predicted_idx
    ]

    scores = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return predicted_class, scores


# ==========================================================
# GRADIO
# ==========================================================

title = " Classification des contenus Netflix avec BERT"

description = """
Classification automatique des résumés Netflix.

Le modèle BERT a été fine-tuné pour prédire
si un contenu Netflix est :

• Movie

• Series

Saisissez un résumé et cliquez sur Submit.
"""

examples = [
    [
        """
        A detective investigates a series
        of mysterious murders in New York.
        """
    ],
    [
        """
        A family navigates life, love and
        friendship across multiple seasons.
        """
    ]
]

demo = gr.Interface(
    fn=predict,

    inputs=gr.Textbox(
        lines=8,
        placeholder="Entrez un résumé Netflix..."
    ),

    outputs=[
        gr.Label(label="Classe prédite"),
        gr.Label(label="Probabilités")
    ],

    title=title,

    description=description,

    examples=examples,
)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True
)
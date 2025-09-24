import json
import torch
import numpy as np

from datasets import Dataset
from setfit import SetFitModel
from sklearn.metrics import f1_score, accuracy_score

with open("../data/validate_skills.json", "r") as f:
    validation_data = json.load(f)

# skill categories (lowercase to match with validate_skills.json)
SKILL_CATEGORIES = [
    "empathy", "validation", "suggestion", "question",
    "professionalism", "self-disclosure", "structure"
]

# load fine-tuned model
model_path = "../finetune/best_setfit_deberta_.560"
model = SetFitModel.from_pretrained(model_path)

texts, labels = [], []

for example in validation_data:
    # extract original and alternative responses
    original_response = " ".join([msg for msg in example["input"] if "Helper:" in msg])
    alternative_response = example["annotations"].get("alternative", "")

    # extract skill labels for original and alternative
    original_labels = [int(example["annotations"].get(f"original-has{skill.lower()}", 0)) for skill in SKILL_CATEGORIES]
    alternative_labels = [int(example["annotations"].get(f"alternative-has{skill.lower()}", 0)) for skill in SKILL_CATEGORIES]

    texts.append(original_response)
    labels.append(original_labels)

    if alternative_response.strip():  
        texts.append(alternative_response)
        labels.append(alternative_labels)

texts = np.array(texts)
labels = np.asarray(labels, dtype=int)

if labels.ndim == 1:
    labels = labels.reshape(-1, len(SKILL_CATEGORIES))


y_pred = model.predict(texts)


if y_pred.ndim == 1 or y_pred.shape[1] != len(SKILL_CATEGORIES):
    print(f"Warning: Model output shape is {y_pred.shape}, expected {(len(labels), len(SKILL_CATEGORIES))}.")
    
    y_pred = np.tile(y_pred.reshape(-1, 1), (1, len(SKILL_CATEGORIES)))  

y_pred_binary = (y_pred >= 0.5).astype(int)

multi_label_f1 = f1_score(labels, y_pred_binary, average="macro")
overall_accuracy = accuracy_score(labels.flatten(), y_pred_binary.flatten())

per_skill_f1 = f1_score(labels, y_pred_binary, average=None)
skill_f1_scores = {skill: round(score, 4) for skill, score in zip(SKILL_CATEGORIES, per_skill_f1)}

per_skill_accuracy = {
    skill: round(accuracy_score(labels[:, i], y_pred_binary[:, i]), 4)
    for i, skill in enumerate(SKILL_CATEGORIES)
}

results = {
    "multi_label_f1_score": round(multi_label_f1, 4),
    "overall_accuracy": round(overall_accuracy, 4),
    "per_skill_f1_scores": skill_f1_scores,
    "per_skill_accuracy": per_skill_accuracy
}

output_path = "./validation_results.json"

with open(output_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\n Validation results saved to {output_path}")

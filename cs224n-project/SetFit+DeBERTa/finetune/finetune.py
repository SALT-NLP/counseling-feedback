import json
import os
import torch
import numpy as np

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from transformers import AutoModel
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE, RandomOverSampler

with open("../data/balanced_updated_converted_train.json", "r") as f:
    data = json.load(f)

SKILL_CATEGORIES = [
    "Empathy", "Validation", "Suggestions", "Questions",
    "Professionalism", "Self-disclosure", "Structure"
]

texts, labels = [], []
for example in data:
    helper_responses = " ".join([msg for msg in example["input"] if "Helper:" in msg])
    label_vector = [skill for skill in example["annotations"]["goodareas"] if skill in SKILL_CATEGORIES]

    texts.append(helper_responses)
    labels.append(label_vector)

mlb = MultiLabelBinarizer(classes=SKILL_CATEGORIES)
labels_array = mlb.fit_transform(labels)

texts_resampled, labels_resampled = [], []
ros = RandomOverSampler()

for i in range(labels_array.shape[1]):  
    ros_i = RandomOverSampler()
    texts_i, labels_i = ros_i.fit_resample(np.array(texts).reshape(-1, 1), labels_array[:, i])
    
    texts_resampled.extend(texts_i.flatten())
    labels_resampled.extend(labels_i.reshape(-1, 1))

texts_resampled = np.array(texts_resampled)
labels_resampled = np.hstack(labels_resampled)  


dataset = Dataset.from_dict({"text": texts_resampled.tolist(), "label": labels_resampled.tolist()})

train_test_split_ratio = 0.8
train_size = int(len(dataset) * train_test_split_ratio)
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

model_name = "microsoft/deberta-v3-small"
model = SetFitModel.from_pretrained(model_name)

# reduce VRAM usage
transformer_model = model.model_body[0].auto_model  
transformer_model.config.gradient_checkpointing = True  

model.model_head = LogisticRegression(max_iter=1000)

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="f1",
    batch_size=6,  # was at 8 but redlining gpu memory, so moved to 6
    num_iterations=10,  # reduced to optimize training, after confirming functionality increase this to ~15 for better results
    column_mapping={"text": "text", "label": "label"}
)

best_f1_score = 0
best_model_path = "./best_setfit_deberta"


early_stopping_patience = 3
epochs_without_improvement = 0


for epoch in range(20):  
    trainer.train()  

    y_pred = trainer.model.predict(test_dataset["text"])
    y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)  
    y_true = np.array(test_dataset["label"])

    multi_label_f1 = f1_score(y_true, y_pred_binary, average="macro")  
    print(f"epoch {epoch + 1}: multi-label f1 score = {multi_label_f1:.4f}")

    if multi_label_f1 > best_f1_score:
        best_f1_score = multi_label_f1
        epochs_without_improvement = 0
        trainer.model.save_pretrained(best_model_path)
        print(f"new best model saved at {best_model_path} (f1 score: {best_f1_score:.4f})")
    else:
        epochs_without_improvement += 1
    
    # add early stopping
    if epochs_without_improvement >= early_stopping_patience:
        print("early stopping triggered. no improvement in last epochs.")
        break

print(f"best model saved at {best_model_path} with f1 score {best_f1_score:.4f}")

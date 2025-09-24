import modal
import os

app = modal.App("fine-tune-setfit")

# define persistent modal volume
MODEL_VOLUME = modal.Volume.from_name("trained-models", create_if_missing=True)

# dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "numpy",
        "setfit",
        "scikit-learn",
        "datasets",
        "imbalanced-learn",
        "tiktoken",
        "sentencepiece",
    )
    .add_local_dir(".", remote_path="/root/data", ignore=["venv", "__pycache__"])  
)

# using H100
@app.function(image=image, gpu="H100", timeout=86400, volumes={"/root/models": MODEL_VOLUME})
def train_model():
    print("Files inside /root/data:")
    print(os.listdir("/root/data"))

    import json
    import torch
    import numpy as np
    from datasets import Dataset
    from setfit import SetFitModel, SetFitTrainer
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MultiLabelBinarizer
    from imblearn.over_sampling import RandomOverSampler

    dataset_path = "/root/data/balanced_updated_converted_train.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)

    
    SKILL_CATEGORIES = [
        "Empathy", "Validation", "Suggestions", "Questions",
        "Professionalism", "Self-disclosure", "Structure",
    ]

    
    texts, labels = [], []
    for example in data:
        helper_responses = " ".join([msg for msg in example["input"] if "Helper:" in msg])
        label_vector = [skill for skill in example["annotations"]["goodareas"] if skill in SKILL_CATEGORIES]

        texts.append(helper_responses)
        labels.append(label_vector)

    # convert labels into multi-hot encoded format
    mlb = MultiLabelBinarizer(classes=SKILL_CATEGORIES)
    labels_array = mlb.fit_transform(labels)

    # apply oversampling on less seen skills
    texts_resampled, labels_resampled = [], []

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

    # load SetFit model with DeBERTa-base
    model_name = "microsoft/deberta-v3-base"
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
        batch_size=6,  # adjust for GPU memory, 8 gives us 9 it/s, 4 gives us 13 it/s, 6 gives us 10 it/s
        num_iterations=10,  # change this for more training -- changed to 10 for compute cost on h100
        column_mapping={"text": "text", "label": "label"},
    )

    best_f1_score = 0
    best_model_path = ""  

    
    early_stopping_patience = 3
    epochs_without_improvement = 0

    
    for epoch in range(15):  
        print(f"Training epoch {epoch + 1}...")
        trainer.train()

        # evaluate on test set
        y_pred = trainer.model.predict(test_dataset["text"])
        y_pred_binary = (np.array(y_pred) >= 0.5).astype(int)
        y_true = np.array(test_dataset["label"])

        # compute Multi-Label F1-score
        multi_label_f1 = f1_score(y_true, y_pred_binary, average="macro")
        print(f"Epoch {epoch + 1}: Multi-Label F1 Score = {multi_label_f1:.4f}")

        f1_str = f"{multi_label_f1:.4f}".replace(".", "_")  
        best_model_path = f"/root/models/best_setfit_deberta_f1_{f1_str}"  

        if multi_label_f1 > best_f1_score:
            best_f1_score = multi_label_f1
            epochs_without_improvement = 0

            trainer.model.save_pretrained(best_model_path)

            print(f"New best model saved at {best_model_path} (F1 Score: {best_f1_score:.4f})")
        else:
            epochs_without_improvement += 1

        # early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered. No improvement in last epochs.")
            break

    print(f"Best model saved at {best_model_path} with F1 Score {best_f1_score:.4f}")

    return best_model_path


@app.local_entrypoint()
def main():
    
    best_model_path = train_model.remote()

    model_filename = os.path.basename(best_model_path)
    
    local_model_path = f"./{model_filename}"

    print(f"Downloading trained model directory from Modal Volume to {local_model_path}...")
    print(f"Model successfully downloaded to {local_model_path}")

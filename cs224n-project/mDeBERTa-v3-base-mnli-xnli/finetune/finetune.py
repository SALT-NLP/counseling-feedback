import json
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Define skill categories
SKILL_CATEGORIES = ["Empathy", "Validation", "Suggestions", "Questions", "Professionalism", "Self-disclosure", "Structure"]

def load_and_format_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_data = []
    
    for example in raw_data:
        text = example["text"]

        if "### Response:" in text:
            text = text.split("### Response:")[0].strip()

        try:
            response_part = example["text"].split("### Response:")[1].strip()
            response_json = json.loads(response_part) 
        except (IndexError, json.JSONDecodeError):
            print(f"Skipping malformed entry: {text[:50]}")
            continue  
        
        # extract good and bad areas
        good_areas = response_json.get("goodareas", [])  
        bad_areas = response_json.get("badareas", [])  

        # create binary multi-label vector 
        label_dict = {
            skill: (1.0 if skill in good_areas else (0.2 if skill in bad_areas else 0.0))
            for skill in SKILL_CATEGORIES
        }

        label_list = [label_dict[skill] for skill in SKILL_CATEGORIES]

        formatted_data.append({
            "text": text,
            "labels": label_list
        })

    return Dataset.from_list(formatted_data)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)


# Main Driver Code -----------------------

# load and format data
dataset = load_and_format_data("../train.json")

# Split dataset further (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

# load tokenizer (https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    num_labels=len(SKILL_CATEGORIES),
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True # original model is trained only for 3 labels
)

# specify which directory to save model
save_directory = "./fine_tuned_mDeBERTa"

training_args = TrainingArguments(
    output_dir=save_directory,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,  
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

trainer.train()

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"\u2705 Model & tokenizer saved to {save_directory}")
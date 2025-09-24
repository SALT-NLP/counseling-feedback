import json
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

skill = "Reflection"
SKILL_CATEGORIES = ["Empathy", "Reflection", "Validation", "Suggestions", "Questions", "Professionalism", "Self-disclosure", "Structure"]

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# load data
dataset_id = "youralien/feedback_qesconv_16wayclassification"
dataset = load_dataset(dataset_id, split="train")

# Split dataset further (80%) and validation (20%)
split_dataset = dataset.train_test_split(test_size=0.2)
# print(f"Train dataset size: {len(split_dataset['train'])}")
# print(f"Test dataset size: {len(split_dataset['test'])}")
# train_dataset = split_dataset['train']
# eval_dataset = split_dataset['test']

# apply tokenization
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model_id = "meta-llama/Llama-3.2-1b"
model_name = "Llama3.2-1b" # "Llama2-7b"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.pad_token_id = tokenizer.eos_token_id
# load model
model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    num_labels=2,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True # original model is trained only for 3 labels
)

# which_class = "Empathy-goodareas"
#which_class = "Questions-badareas" # "Suggestions-badareas" # "Reflections-badareas" # "Empathy-badareas"
# SKILL_OPTIONS = ["Reflections", "Validation", "Empathy", "Questions", "Suggestions", "Self-disclosure", "Structure", "Professionalism"]
# goodareas_to_ignore = [f"{skill}-goodareas" for skill in SKILL_OPTIONS if f"{skill}-goodareas" != which_class]
# badareas_to_ignore = [f"{skill}-badareas" for skill in SKILL_OPTIONS if f"{skill}-badareas" != which_class]
# cols_to_remove = ['conv_index', 'helper_index', 'input', 'text']
# cols_to_remove.extend(goodareas_to_ignore)
# cols_to_remove.extend(badareas_to_ignore)
# if which_class in split_dataset["train"].features.keys():
classifier_name = f"{skill}-badareas-suboptimal" #"{skill}-badareas-shouldHave" "{skill}-badareas-shouldNotHave"
split_dataset =  split_dataset.rename_column(classifier_name, "labels") # to match Trainer
tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

save_directory = f"{model_name}-{classifier_name}-classifier"
training_args = TrainingArguments(
    output_dir=save_directory,
    eval_strategy="epoch",
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

trainer.train()
# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

tokenizer.save_pretrained(save_directory)
print("after trainer")
trainer.create_model_card()
trainer.push_to_hub()

print(f"\u2705 Model & tokenizer saved")
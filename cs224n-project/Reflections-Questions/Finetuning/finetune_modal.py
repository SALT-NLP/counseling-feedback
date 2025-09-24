import modal
import os

# Define Modal app
app = modal.App("badareas-classifier")

image = ( 
  modal.Image.from_registry("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
  .pip_install_from_requirements("requirements.txt") 
)

@app.function(
        image=image,
        gpu="A100-80GB", 
        timeout=86400,
        secrets=[modal.Secret.from_name("huggingface-secret")])
def train_model():
    from huggingface_hub import login, HfFolder
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback #BitsAndBytesConfig,
    from huggingface_hub import HfFolder
    import numpy as np
    #from sklearn.metrics import f1_score
    from transformers.utils import logging
    #from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    #from trl import SFTTrainer
    import torch
    import evaluate

    # Login to Hugging Face
    print("HF_TOKEN:", os.getenv("HF_TOKEN"))
    login(token= os.environ["HF_TOKEN"])
    print("Logged into Hugging Face.")

    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    logger.info("LOGGER")

    skill = "Reflection"

    def tokenize_function(example):
        # IDK WHAT THE TABLE LOOKS LIKE BUT  
        # !!!!!!!! "example["text"]" should be the last helper prompt !!!!!!!!!!
        print("Within tokenize_function: ": example["entry"]["input"][-1].removeprefix("Helper: "))
        return tokenizer(example["entry"]["input"][-1].removeprefix("Helper: "), padding="max_length", truncation=True, max_length=512)

    # load data
    dataset_id = "huangfe/badareas_augmented_dataset_reflections_questions" #"youralien/feedback_qesconv_16wayclassification" # replace with our dataset
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

    model_id = "bert-base-cased"
    model_name = "bert" # "Llama2-7b"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # load model    
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, torch_dtype="auto")

    classifier_name = f"{skill}-badareas-suboptimal" #"{skill}-badareas-shouldHave" "{skill}-badareas-shouldNotHave"
    
    # !!!!! CHANGE THESE COLUMN NAMES BASED ON THE TABLE COLUMN NAMES !!!!!
    suffixes = ["-badareas-suboptimal", "-badareas-shouldhave", "-badareas-shouldnothave" ]
               #"Validation-badareas", "Empathy-badareas", "Questions-badareas", "Suggestions-badareas", "Self-disclosure-badareas", "Structure-badareas", "Professionalism-badareas"]
    skill_categories = [
        "empathy", "validation", "suggestion", "question",
        "professionalism", "self-disclosure", "structure"
    ]
    columns = []
    for s in skill_categories:
        for suffix in suffixes:
            columns.append(s+suffix)
    print("all columns: ", columns)

    columns_to_remove = [f"{c}" for c in columns if c != classifier_name]
    cols_to_remove = ['conv_index', 'helper_index', 'input', 'text']
    cols_to_remove.extend(columns_to_remove)
    print("columns to remove: ", columns)
    # if which_class in split_dataset["train"].features.keys():
    split_dataset =  split_dataset.rename_column(classifier_name, "labels") # to match Trainer
    tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)
    #check 
    print("Example from Tokenized_dataset: ", tokenized_dataset["train"][0])
    print("All Features: ", tokenized_dataset["train"].features)

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

    print("Model & tokenizer saved")

if __name__ == "__main__":
    with app.run():
        train_model()
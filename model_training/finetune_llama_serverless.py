"""
Fine-tune LLaMA model using Modal for serverless execution.

To run this script, ensure you have Modal installed and configured. 
Make sure you are in the model_training directory. Then run
```bash
modal run finetune_llama_serverless.py
```
Assumes the train/test datasets have been already preprocessed and is available locally at model_training/data/feedback_qes_conv_processed/
"""
import modal
import os
from dataclasses import dataclass, field
from typing import Optional

app = modal.App("finetune-multilevelfeedback-llama")

# Define persistent modal volume for trained models
MODEL_VOLUME = modal.Volume.from_name("trained-llama-models", create_if_missing=True)

# Define dependencies - similar to the original script but adapted for Modal
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
        "trl",
        "bitsandbytes",
        "python-dotenv",
        "numpy",
        "scipy",
    )
    .add_local_dir(".", remote_path="/root/model_training", ignore=["venv", "__pycache__", "*.git*"])
)

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B", metadata={"help": "the model name"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    dataset_name: Optional[str] = field(
        default="feedback_qes_conv_processed", metadata={"help": "dataset name"}
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # Training arguments
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(default=4, metadata={"help": "gradient accumulation steps"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "warmup steps"})
    num_train_epochs: Optional[float] = field(default=3.0, metadata={"help": "number of training epochs"})
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "learning rate"})
    optim: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "optimizer"})
    weight_decay: Optional[float] = field(default=0.001, metadata={"help": "weight decay"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "learning rate scheduler"})
    output_dir: Optional[str] = field(default="/root/model_training/models/output", metadata={"help": "SFT output directory"})
    report_to: Optional[str] = field(default="none", metadata={"help": "reporting"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "gradient checkpointing"})

    # HF Token
    hf_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace token"})


# Using H100 GPU like in the SetFit example
@app.function(
    image=image,
    gpu="H100",
    timeout=86400,  # 24 hours
    volumes={"/root/model_training/models": MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")]  # Assumes HF token is stored as secret
)
def train_llama_model(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    dataset_name: str = "feedback_qes_conv_processed",
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
):
    print("Files inside /root/model_training:")
    print(os.listdir("/root/model_training"))

    import torch
    from transformers import (
        BitsAndBytesConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        TrainingArguments,
        set_seed
    )
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, AutoPeftModelForCausalLM

    # Set seed for reproducibility
    set_seed(42)

    # Get HF token from Modal secret
    access_token = os.environ.get("HUGGINGFACE_TOKEN")

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    current_device = torch.cuda.current_device()
    print("Current GPU: ", current_device)

    # Load the model config and adjust rope_scaling
    config = AutoConfig.from_pretrained(model_name, token=access_token)
    config.rope_scaling = {
        "type": "linear",
        "factor": 8.0
    }

    # PEFT configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("---------------------------------")
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    print("---------------------------------")

    # Load dataset
    try:
        dataset = load_dataset(
            "json",
            data_files={
                "train": f"/root/model_training/data/{dataset_name}/train.json",
                "test": f"/root/model_training/data/{dataset_name}/test.json"
            },
            split={"train": "train", "test": "test"},
            cache_dir="/root/model_training/cache"
        )
        print(f"Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"FAILED TO LOAD DATASET: {e}")
        return None

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # Format dataset
    dataset['train'] = dataset['train'].map(lambda x: {'text': f'<s>{x["text"]}</s>'})

    print("Printing length of train dataset: ", len(dataset['train']))
    print('---------------------------------')
    print("Sample text:")
    print(dataset['train'][0]['text'])
    print('---------------------------------')

    # BitsAndBytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        token=access_token,
    )

    # Enable gradient checkpointing for memory efficiency
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # Create training arguments
    training_args = TrainingArguments(
        output_dir="/root/model_training/models/output",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        optim="paged_adamw_32bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        report_to="none",
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            output_dir=training_args.output_dir,
            report_to=training_args.report_to,
            dataset_num_proc=4,
            packing=True,
            dataset_text_field="text",
            logging_steps=training_args.logging_steps,
            save_steps=training_args.save_steps,
        ),
    )

    # Train the model
    print("Starting training...")
    results = trainer.train()
    print("Training results:", results)

    # Save the trained model
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del base_model
    torch.cuda.empty_cache()

    print("Loading model for merging...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Merge and unload LoRA weights
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")

    # Save the final merged model
    try:
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_merged_dir)
        print(f"Model successfully saved to {output_merged_dir}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

    print("TRAINING COMPLETE")
    return output_merged_dir


@app.local_entrypoint()
def main(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    dataset_name: str = "feedback_qes_conv_processed",
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
):
    print(f"Starting LLaMA fine-tuning with model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Training epochs: {num_train_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {per_device_train_batch_size}")
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")

    # Run the training
    model_path = train_llama_model.remote(
        model_name=model_name,
        dataset_name=dataset_name,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    if model_path:
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
    else:
        print("Training failed!")
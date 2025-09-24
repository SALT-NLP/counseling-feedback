import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (BitsAndBytesConfig, HfArgumentParser, TrainingArguments,
                          AutoModelForCausalLM, AutoTokenizer, AutoConfig)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

from transformers import set_seed
set_seed(42)

load_dotenv()

access_token = os.getenv("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_device = torch.cuda.current_device()
print("Current GPU: ", current_device)

@dataclass
class ScriptArguments:
    # model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B", metadata={"help": "the model name"})

    # error resolved when packing is True https://github.com/huggingface/transformers/issues/15505#issuecomment-2220822670
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    dataset_name: Optional[str] = field(default="feedback_qesconv", metadata={"help": "dataset name"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # add optional checkpoint
    checkpoint_path: Optional[str] = field(default=None, metadata={"help": "if checkpoint has been saved before -- use it"})
    

parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

# load the model config and adjust rope_scaling
config = AutoConfig.from_pretrained(script_args.model_name)
config.rope_scaling = {
    "type": "linear",
    "factor": 8.0
}

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print("---------------------------------")
print(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, token=access_token)
print("---------------------------------")


# set dataset
try:
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"data/{script_args.dataset_name}/train.json",
            "test": f"data/{script_args.dataset_name}/test.json"
        },
        split={"train": "train", "test": "test"},
        cache_dir="./cache"
    )
except Exception as e:
    print(f"FAILED TO LOAD DATASET: {e}")

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = 'right'
dataset['train'] = dataset['train'].map(lambda x: {'text': f'<s>{x["text"]}</s>'})


print("Printing length of train dataset: ", len(dataset['train']))
print('---------------------------------')
print(dataset['train'][0]['text'])
print('---------------------------------')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
)

base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        config=config,
        quantization_config=bnb_config,
        device_map="auto",
        token=access_token,
)

# increases memory pressure -- so turned off for now (pass into command as False)
if training_args.gradient_checkpointing:
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False


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
        packing=script_args.packing,
        dataset_text_field="text", 
        resume_from_checkpoint=script_args.checkpoint_path
    ),
)

results = trainer.train()
print(results)

trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)

model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")

# ensure model is properly saved
if os.path.exists(output_merged_dir):
    print(f"The directory '{output_merged_dir}' exists.")
    
    # needed files for a complete model save 
    model_files = ['pytorch_model.bin', 'config.json', 'special_tokens_map.json']
    
    missing_files = [file for file in model_files if not os.path.exists(os.path.join(output_merged_dir, file))]
    
    if missing_files:
        print(f"Missing the following model files: {', '.join(missing_files)}")
    else:
        print("All expected model files are present.")
else:
    print(f"The directory '{output_merged_dir}' does not exist.")

# error handling to save model
try:
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    print(f"Model successfully saved to {output_merged_dir}")
except Exception as e:
    print(f"Error saving model: {e}")

print("PROGRAM COMPLETE")


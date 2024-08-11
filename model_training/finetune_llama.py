import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (BitsAndBytesConfig, HfArgumentParser, TrainingArguments,
                          AutoModelForCausalLM, AutoTokenizer)
from datasets import load_dataset
from trl import SFTTrainer

from transformers import set_seed
set_seed(42)

access_token = os.environ.get("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ScriptArguments:
    # model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", metadata={"help": "the model name"})

    # error resolved when packing is True https://github.com/huggingface/transformers/issues/15505#issuecomment-2220822670
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    dataset_name: Optional[str] = field(default="feedback_qesconv", metadata={"help": "dataset name"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

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
print(script_args.dataset_name)
dataset = load_dataset(f"data/{script_args.dataset_name}")

tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = 'right'
dataset['train'] = dataset['train'].map(lambda x: {'text': f'<s>{x["text"]}</s>'})


print(len(dataset['train']))
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
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    token=access_token
)

if training_args.gradient_checkpointing:
    base_model.gradient_checkpointing_enable()

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    max_seq_length=2048,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
    packing=script_args.packing,
)

results = trainer.train()
print(results)

trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)



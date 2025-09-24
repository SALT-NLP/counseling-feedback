import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig

from trl import DPOTrainer, DPOConfig

from transformers import set_seed
set_seed(42)
torch.manual_seed(42)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: str = field(
        metadata={"help": "the location of the SFT model name or path"},
    )
    output_dir: str = field(
            metadata={"help": "the output directory"}
    )
    # data parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})
    # training parameters
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=1024, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10, metadata={"help": "the evaluation frequency"})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

# additions for debugging purposes
MODEL_NAME = "meta-llama/Llama-3.1-8B"


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset("data/feedback_qesconv_dpo")
    original_columns = dataset.column_names


    dataset['train'] = dataset['train'].map(lambda x: {'prompt': f'<s>{x["prompt"]}', 'chosen': f'{x["chosen"]}</s>', 'rejected': f'{x["rejected"]}</s>'})
    dataset['test'] = dataset['test'].map(lambda x: {'prompt': f'<s>{x["prompt"]}', 'chosen': f'{x["chosen"]}</s>', 'rejected': f'{x["rejected"]}</s>'})

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # 4. initialize training arguments:
    dpo_config = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True, 
        remove_unused_columns=False,
        run_name="dpo_model",
        # DPO-specific parameters
        beta=script_args.beta,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint_dpo")
    dpo_trainer.model.save_pretrained(output_dir)

    del model
    del model_ref
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint_dpo")
    model.save_pretrained(output_merged_dir, safe_serialization=True)


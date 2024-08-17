# Model training & sampling

### Fine-tuning the model

```
poetry run accelerate launch finetune_llama.py \
--output_dir="<SFT PATH>" \
--max_steps=766 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=3e-4 \
--lr_scheduler_type="cosine" \
--warmup_steps=100 \
--weight_decay=0.01 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="sft_model" \
--report_to="wandb"
```

#### Generating 10 samples from the model and scoring

```
poetry run accelerate launch generate_llama_exp.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv_dpo_pre" \
--start_index=0

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_dpo_pre_<SFT PATH>_generations_0.json"

```


### DPO alignment

```
poetry run accelerate launch dpo_llama.py  \
--model_name_or_path="<SFT PATH/final_merged_checkpoint>" \
--output_dir="<DPO PATH>" \

```


### Ablations

##### + new data training

```
poetry run accelerate launch generate_llama.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv_dpo_pre" 
```

```
poetry run accelerate launch finetune_llama.py \
--model_name="<SFT PATH>" \
--output_dir="<NEW DATA PATH>" \
--dataset_name="feedback_ablation_data" \
--max_steps=200 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=5e-6 \
--lr_scheduler_type="cosine" \
--warmup_steps=10 \
--weight_decay=0.05 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="ablation_new_data" \
--report_to="wandb"
```

##### + best scores training

```
poetry run accelerate launch finetune_llama.py \
--model_name="<SFT PATH>" \
--output_dir="<BEST SCORES PATH>" \
--dataset_name="feedback_ablation_preference" \
--max_steps=200 \
--logging_steps=10 \
--save_steps=100 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--gradient_checkpointing=False \
--group_by_length=False \
--learning_rate=5e-6 \
--lr_scheduler_type="cosine" \
--warmup_steps=10 \
--weight_decay=0.05 \
--optim="paged_adamw_32bit" \
--bf16=True \
--remove_unused_columns=False \
--run_name="ablation_best_scores" \
--report_to="wandb"
```
##### sampling and scoring with SFT
```

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<NEW DATA PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<BEST SCORES PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

poetry run accelerate launch generate_llama_exp.py \
--output_dir="<DPO PATH>" \
--bf16=True \
--dataset_name="feedback_qesconv" 

```

```

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_sft_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_dpo_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_ablation_preference_model_generations_0.json"

poetry run accelerate launch generate_llama_exp_probs.py \
--output_dir="<SFT PATH>" \
--bf16=True \
--file_name="exp/feedback_qesconv_ablation_data_model_generations_0.json"

```

We used a single A100 GPU.


### Acknowledgements

We use TRL - Transformer Reinforcement Learning library [TRL](https://github.com/huggingface/trl) and build upon provided implementation [examples](https://github.com/huggingface/trl/tree/main/examples/research_projects)

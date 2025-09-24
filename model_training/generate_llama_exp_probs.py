import os
from dataclasses import dataclass, field
from typing import Optional
import json

from tqdm import tqdm

import torch
from transformers import HfArgumentParser, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from torch.nn.functional import softmax

from transformers import set_seed
set_seed(42)
torch.manual_seed(42)

MAX_TRIES = 5

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29913, 12258, 500, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

access_token = os.environ.get("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B", metadata={"help": "the model name"})
    file_name: Optional[str] = field(default="", metadata={"help": "file name"})

parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, token=access_token)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# read exp/koko.json
with open(script_args.file_name) as f:
    dataset = json.load(f)

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)

model.eval()

def get_probability_of_true(model, tokenizer, prompt):
    new_prompt_encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**new_prompt_encoded)
        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    probabilities = softmax(last_token_logits, dim=0)
    
    # Get the token ID for 'true' (without the special character)
    t_index = tokenizer.convert_tokens_to_ids('true')
    print(f"t_index: {t_index}")

    # Check if the token exists in the vocabulary
    if t_index == tokenizer.unk_token_id:
        print("Warning: 'true' token not found in vocabulary")
        probability_of_t = 0
    else:
        # Make sure t_index is within the range of the probabilities tensor
        if t_index < len(probabilities):
            probability_of_t = probabilities[t_index].item()
        else:
            print(f"Warning: t_index ({t_index}) is out of range for probabilities tensor (length {len(probabilities)})")
            probability_of_t = 0

    print(f"Probability of 'true': {probability_of_t}")
    return probability_of_t

for ind in tqdm(range(len(dataset))):
    original_feedback = '<s>' + dataset[ind]['prompt'] + json.dumps(dataset[ind]['output'][0]).split('perfect\": ')[0] + 'perfect\":'
    dataset[ind]['prob'] = get_probability_of_true(model, tokenizer, original_feedback)

    for output in dataset[ind]['output']:
        if 'improved' in output:
            splitted = dataset[ind]['prompt'].split('Response:')[0].split('\n')
            splitted = splitted[:-2]
            splitted[-1] = f'Helper: {output["alternative"]}'
            new_prompt = '\n'.join(splitted) + '\n\n### Response:'

            # add perfect
            improved_feedback = json.dumps(output['improved']).split('perfect\": ')[0] + 'perfect\":'
            new_prompt = '<s>' + new_prompt + improved_feedback

            output['improved']['prob'] = get_probability_of_true(model, tokenizer, new_prompt)

model_name = training_args.output_dir.split('/')[-1]
# save dataset to exp_probs.json
with open(f'{script_args.file_name.split(".")[0]}_{model_name}_probs.json', "w") as outfile:
    json.dump(dataset, outfile)

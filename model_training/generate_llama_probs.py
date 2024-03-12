import os
from dataclasses import dataclass, field
from typing import Optional
import json

from tqdm import tqdm

import torch
from transformers import  HfArgumentParser, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from torch.nn.functional import softmax

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
    model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
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


for ind in tqdm(range(len(dataset))):

    output = dataset[ind]['output']

    # print("\n\noriginal prompt")
    # print(dataset[ind]['prompt'])
    # print("-------------------")

    splitted = dataset[ind]['prompt'].split('Response:')[0].split('\n')
    splitted = splitted[:-2]
    if 'alternative' in output:
        splitted[-1] = f'Helper: {output["alternative"]}'
        new_prompt = '\n'.join(splitted) + '\n\n### Response:'

        helper_line = new_prompt.split('Response:')[0].split('\n')[-3]

        new_prompt = new_prompt + json.dumps({"perfect":True})[:-6]

        # print("modified prompt")
        # print(new_prompt)

        new_prompt_encoded = tokenizer(new_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**new_prompt_encoded)
            logits = outputs.logits

        last_token_logits = logits[0, -1, :]
        probabilities = softmax(last_token_logits, dim=0)
        max_prob_index = torch.argmax(probabilities).item()
        predicted_token = tokenizer.convert_ids_to_tokens(max_prob_index)
        t_index = tokenizer.convert_tokens_to_ids('▁true')
        probability_of_t = probabilities[t_index].item()

        output['improved_prob'] = probability_of_t

        # print(output['improved_prob'])

model_name = training_args.output_dir.split('/')[-1]

with open(f'{script_args.file_name.split(".")[0]}_scored_by_{model_name}_probs.json', "w") as outfile:
    json.dump(dataset, outfile)


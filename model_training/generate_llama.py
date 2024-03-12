import os
from dataclasses import dataclass, field
from typing import Optional
import json

from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from transformers import HfArgumentParser, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from transformers import set_seed
set_seed(42)
torch.manual_seed(42)

MAX_TRIES = 5


def ann_check(ann):
    # if "helper" not in ann:
    #     raise Exception("No helper in annotation!")
    if "goodareas" not in ann:
        raise Exception("No goodareas in annotation!")
    if "perfect" not in ann:
        raise Exception("No perfect in annotation!")
    if ann["perfect"] == False:
        if "feedback" not in ann:
            raise Exception("No feedback in annotation!")
        if "badareas" not in ann:
            raise Exception("No areas in annotation!")
        if "alternative" not in ann:
            raise Exception("No alternative in annotation!")


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
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})
    start_index: Optional[int] = field(default=0, metadata={"help": "start index"})
    dataset_name: Optional[str] = field(default="feedback_qesconv", metadata={"help": "dataset name"})
    threshold: Optional[int] = field(default=0.5, metadata={"help": "threshold"})


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, token=access_token)

dataset = load_dataset(f"data/{script_args.dataset_name}")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

dataset['test'] = dataset['test'].map(lambda x: {'text': f'<s>{x["text"]}',
                                                 'helper_index': x['helper_index'],
                                                 'conv_index': x['conv_index']})
dpo = ""
if 'dpo' in training_args.output_dir:
    dpo = "_dpo"
output_merged_dir = os.path.join(training_args.output_dir, f"final_merged_checkpoint{dpo}")
print(f"Loading: {output_merged_dir}")
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)

model.eval()


def extract_output(s):
    start_index = s.find("Response:")
    start_index += len("Response:")
    extracted_string = s[start_index:]#.strip()
    return extracted_string


with torch.no_grad():
    generations = []
    for ind in tqdm(range(script_args.start_index, len(dataset['test']))):

        helper_line = dataset['test'][ind]['text'].split('Response:')[0].split('\n')[-3]

        original_feedback = dataset['test'][ind]['text'] + json.dumps({"perfect":True})[:-6]

        new_prompt_encoded = tokenizer(original_feedback, add_special_tokens=False, return_tensors="pt").to(model.device)

        outputs = model(**new_prompt_encoded)
        logits = outputs.logits

        last_token_logits = logits[0, -1, :]
        probabilities = softmax(last_token_logits, dim=0)
        max_prob_index = torch.argmax(probabilities).item()
        predicted_token = tokenizer.convert_ids_to_tokens(max_prob_index)
        t_index = tokenizer.convert_tokens_to_ids('▁true')
        probability_of_t = probabilities[t_index].item()

        threshold = script_args.threshold
        if probability_of_t > threshold:
            # return true
            feedback_to_continue = original_feedback + ' true'
        else:
            feedback_to_continue = original_feedback + ' false'

        query = tokenizer(feedback_to_continue, add_special_tokens=False, return_tensors="pt").to(
            model.device)
        output = model.generate(**query, max_new_tokens=600, do_sample=True, temperature=0.8,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                stopping_criteria=StoppingCriteriaList([StopOnTokens()]))

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        feedback_only = extract_output(decoded_output)


        attempt = 0
        while attempt < MAX_TRIES:
            try:
                loaded_output = json.loads(feedback_only)
                ann_check(loaded_output)

                generations.append(
                    {"prompt": dataset['test'][ind]['text'], "helper_index": dataset['test'][ind]["helper_index"],
                     "conv_index": dataset['test'][ind]["conv_index"], "output": loaded_output,
                     "prob": probability_of_t})

                break

            except Exception as e:
                print(e)
                print(f'### Attempt {attempt} Failed to parse output as json\n\n')
                attempt += 1


        model_name = training_args.output_dir.split('/')[-1]
        # save to json
        with open(f'exp/{script_args.dataset_name}_{model_name}_generations_one.json', "w") as outfile:
            json.dump(generations, outfile)


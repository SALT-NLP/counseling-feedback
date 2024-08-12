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
    # model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", metadata={"help": "the model name"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
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
model = AutoModelForCausalLM.from_pretrained(output_merged_dir, device_map="auto", torch_dtype=torch.bfloat16)

model.eval()

def extract_output(s):
    start_index = s.find("Response:")
    start_index += len("Response:")
    extracted_string = s[start_index:].strip()
    return extracted_string.rstrip('</s>')  # Remove the </s> token

with torch.no_grad():
    generations = []
    for ind in tqdm(range(script_args.start_index, len(dataset['test']))):
        outputs = []
        true_count = 0
        for _ in range(10):

            helper_line = dataset['test'][ind]['text'].split('Response:')[0].split('\n')[-3]

            original_feedback = dataset['test'][ind]['text'] + json.dumps({"perfect": True})[:-6]

            new_prompt_encoded = tokenizer(original_feedback, add_special_tokens=False, return_tensors="pt").to(
                model.device)

            outputs_m = model(**new_prompt_encoded)
            logits = outputs_m.logits

            last_token_logits = logits[0, -1, :]
            probabilities = softmax(last_token_logits, dim=0)
            max_prob_index = torch.argmax(probabilities).item()
            predicted_token = tokenizer.convert_ids_to_tokens(max_prob_index)
            
            # ===== OLD CODE
            t_index = tokenizer.convert_tokens_to_ids('▁true')
            print(f"_t_index: {t_index}")
            print(f"probabilities shape: {probabilities.shape}")

            # ==== OLD LINE THROWING ERROR
            # probability_of_t = probabilities[t_index].item()
                
            # ===== NEW CODE WITH MORE ERROR HANDLING
            # Get the token ID for 'true' (without the special character)
            t_index = tokenizer.convert_tokens_to_ids('true')
            print(f"t_index: {t_index}")

            # Check if the token exists in the vocabulary
            if t_index == tokenizer.unk_token_id:
                print("Warning: 'true' token not found in vocabulary")
                probability_of_t = 0  # or handle this case as appropriate for your use case
            else:
                # Make sure t_index is within the range of the probabilities tensor
                if t_index < len(probabilities):
                    probability_of_t = probabilities[t_index].item()
                else:
                    print(f"Warning: t_index ({t_index}) is out of range for probabilities tensor (length {len(probabilities)})")
                    probability_of_t = 0  # or handle this case as appropriate for your use case

            print(f"Probability of 'true': {probability_of_t}")

            threshold = script_args.threshold
            if probability_of_t > threshold:
                # return true
                feedback_to_continue = original_feedback + ' true'
                label = True
            else:
                feedback_to_continue = original_feedback + ' false'
                label = False

            query = tokenizer(feedback_to_continue, add_special_tokens=False, return_tensors="pt").to(
                model.device)
            output = model.generate(**query, max_new_tokens=600, do_sample=True, temperature=0.8,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    stopping_criteria=StoppingCriteriaList([StopOnTokens()]))

            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            feedback_only = extract_output(decoded_output)

            print("----- feedback only: ", feedback_only)

            attempt = 0
            while attempt < MAX_TRIES:
                try:
                    loaded = json.loads(feedback_only)
                    print("------- loaded feedback: ", loaded)
                    ann_check(loaded)

                    if loaded['perfect']:
                        true_count += 1

                    else:
                        splitted = dataset['test'][ind]['text'].split('Response:')[0].split('\n')
                        splitted = splitted[:-2]
                        print("----- splitted before alt: ", splitted)
                        splitted[-1] = f'Helper: {loaded["alternative"]}'
                        new_prompt = '\n'.join(splitted) + '\n\n### Response:' + json.dumps({"perfect": label})[:-1]
                        print("----- new prompt: ", new_prompt)
                        new_prompt_encoded = tokenizer(new_prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
                        improved_output = model.generate(**new_prompt_encoded, max_new_tokens=600, do_sample=True, temperature=0.8, eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,  stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
                        decoded_improved_output = tokenizer.decode(improved_output[0])

                        try:
                            improved_feedback_only = extract_output(decoded_improved_output)
                            print(" ======= improved_feedback_only ", improved_feedback_only)
                            new_loaded = json.loads(improved_feedback_only)
                            print(" ======= new_loaded", new_loaded)
                            ann_check(new_loaded)
                            print(" ======= ann_check(new_loaded) passed!")
                            loaded["improved"] = new_loaded
                        except:
                            raise Exception("Failed to load alternative")

                    outputs.append(loaded)
                    break

                except Exception as e:
                    print(e)
                    print(f'### Attempt {attempt} Failed to parse output as json\n\n')
                    attempt += 1


        generations.append({"prompt": dataset['test'][ind]['text'], "helper_index": dataset['test'][ind]["helper_index"],
                            "conv_index": dataset['test'][ind]["conv_index"], "output": outputs, "percent": true_count/len(outputs)})

        model_name = training_args.output_dir.split('/')[-1]
    # save to json
        with open(f'exp/{script_args.dataset_name}_{model_name}_generations_{script_args.start_index}.json', "w") as outfile:
            json.dump(generations, outfile)

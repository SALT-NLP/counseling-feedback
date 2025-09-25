"""
Generate samples from fine-tuned LLaMA model using Modal for serverless execution.

To run this script, ensure you have Modal installed and configured.
Make sure you are in the model_training directory. Then run:
```bash
modal run generate_llama_exp_serverless.py --output-dir="<SFT PATH>" --dataset-name="feedback_qesconv" --start-index=0
```

This should be run after finetune_llama_serverless.py has been executed.
Note the output_dir should point to the SFT output directory containing the final_merged_checkpoint.
This is assumed to be at /root/model_training/models/output if using default settings in finetune_llama_serverless.py.
Sorry output is poorly named here, since its the (SFT) output directory, but other models may be used for generation.

Also, assumes the `test.json` dataset is available locally at model_training/data/feedback_qesconv/. This test set does not have a reference response, just the prompt.
This is DIFFERENT from the dataset used for training (`feedback_qes_conv_processed`), which has both prompt and reference response for `train.json` and `valid.json`.
"""
import modal
import os
from dataclasses import dataclass, field
from typing import Optional
import json

app = modal.App("generate-multilevelfeedback-llama")

# Define persistent modal volume for trained models
MODEL_VOLUME = modal.Volume.from_name("trained-llama-models", create_if_missing=True)

# Define dependencies
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
        "tqdm",
    )
    .add_local_dir(".", remote_path="/root/model_training", ignore=["venv", "__pycache__", "*.git*"])
)

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B", metadata={"help": "the model name"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    start_index: Optional[int] = field(default=0, metadata={"help": "start index"})
    dataset_name: Optional[str] = field(default="feedback_qesconv", metadata={"help": "dataset name"})
    threshold: Optional[float] = field(default=0.5, metadata={"help": "threshold"})
    output_dir: Optional[str] = field(default="/root/model_training/models/output", metadata={"help": "output directory where the model checkpoint is located and where experiment generations will be saved"})

MAX_TRIES = 5

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,  # 24 hours
    volumes={"/root/model_training/models": MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_samples(
    output_dir: str = "/root/model_training/models/output",
    dataset_name: str = "feedback_qesconv",
    start_index: int = 0,
    threshold: float = 0.5,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from torch.nn.functional import softmax
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import set_seed

    # Set seed for reproducibility
    set_seed(42)
    torch.manual_seed(42)

    # Get HF token from Modal secret
    access_token = os.environ.get("HUGGINGFACE_TOKEN")

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def ann_check(ann):
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

    def extract_output(s):
        start_index = s.find("Response:")
        start_index += len("Response:")
        extracted_string = s[start_index:].strip()
        return extracted_string.rstrip('</s>')  # Remove the </s> token

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=access_token)

    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(f"/root/model_training/data/{dataset_name}")
        print(f"Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"FAILED TO LOAD DATASET: {e}")
        return None

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    dataset['test'] = dataset['test'].map(lambda x: {'text': f'<s>{x["text"]}',
                                                     'helper_index': x['helper_index'],
                                                     'conv_index': x['conv_index']})

    # Determine model path
    dpo = ""
    if 'dpo' in output_dir:
        dpo = "_dpo"
    output_merged_dir = os.path.join(output_dir, f"final_merged_checkpoint{dpo}")

    print(f"Loading model from: {output_merged_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        output_merged_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    print(f"Starting generation from index {start_index} to {len(dataset['test'])}")

    with torch.no_grad():
        generations = []
        for ind in tqdm(range(start_index, len(dataset['test']))):
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

                # Get the token ID for 'true'
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

                if probability_of_t > threshold:
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

            if len(outputs) == 0:
                print(f"⚠️ ⚠️ ⚠️ Skipping index {ind} due to no valid outputs. ⚠️ ⚠️ ⚠️")
                continue
            
            generations.append({
                "prompt": dataset['test'][ind]['text'],
                "helper_index": dataset['test'][ind]["helper_index"],
                "conv_index": dataset['test'][ind]["conv_index"],
                "output": outputs,
                "percent": true_count/len(outputs)
            })

            model_name = output_dir.split('/')[-1]

            # Save results to the persistent volume
            exp_dir = "/root/model_training/models/exp"
            os.makedirs(exp_dir, exist_ok=True)
            output_file = f'{exp_dir}/{dataset_name}_{model_name}_generations_{start_index}.json'
            with open(output_file, "w") as outfile:
                json.dump(generations, outfile)

            print(f"Saved generations to: {output_file}")

    print("GENERATION COMPLETE")
    return output_file

@app.local_entrypoint()
def main(
    output_dir: str = "/root/model_training/models/output",     # default SFT output directory
    dataset_name: str = "feedback_qesconv",
    start_index: int = 0,
    threshold: float = 0.5,
):
    print(f"Starting LLaMA generation with output_dir: {output_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Start index: {start_index}")
    print(f"Threshold: {threshold}")

    # Run the generation
    result_file = generate_samples.remote(
        output_dir=output_dir,
        dataset_name=dataset_name,
        start_index=start_index,
        threshold=threshold,
    )

    if result_file:
        print(f"Generation completed successfully!")
        print(f"Results saved to: {result_file}")
    else:
        print("Generation failed!")
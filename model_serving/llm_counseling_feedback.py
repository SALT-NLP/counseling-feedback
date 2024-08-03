import os
import json
import time

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from modal import Image, App, method, build, enter, gpu

app = App("llm-counseling-feedback")

MODEL_NAME = "avylor/mh_feedback_model"
N_GPUS = 1  # Heads up: this example has not yet been tested with multiple GPUs
GPU_CONFIG = gpu.A100(count=N_GPUS)

counseling_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers",
        "tqdm"
    )
)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29913, 12258, 500, 2]
        return input_ids[0][-1] in stop_ids

def extract_output(s):
    start_index = s.find("Response:")
    start_index += len("Response:")
    extracted_string = s[start_index:]
    return extracted_string

@app.cls(image=counseling_image, gpu=GPU_CONFIG, container_idle_timeout=300)
class CounselingFeedbackModel:
    @build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_NAME)

    @enter()
    def load_model(self):
        t0 = time.time()
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        self.model.to('cuda')  # Explicitly move model to GPU
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        print(f"Model loaded in {time.time() - t0:.2f}s")

    @method()
    def generate(self, input_text, threshold=0.5):
        t0 = time.time()

        original_feedback = input_text + json.dumps({"perfect":True})[:-6]
        new_prompt_encoded = self.tokenizer(original_feedback, add_special_tokens=False, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = self.model(**new_prompt_encoded)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :]
            probabilities = softmax(last_token_logits, dim=0)
            t_index = self.tokenizer.convert_tokens_to_ids('▁true')
            probability_of_t = probabilities[t_index].item()

        if probability_of_t > threshold:
            feedback_to_continue = original_feedback + ' true'
        else:
            feedback_to_continue = original_feedback + ' false'

        query = self.tokenizer(feedback_to_continue, add_special_tokens=False, return_tensors="pt").to('cuda')

        output = self.model.generate(
            **query,
            max_new_tokens=600,
            do_sample=False,
            # do_sample=True,
            # temperature=0.8,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        feedback_only = extract_output(generated_text)

        print(f"Output generated in {time.time() - t0:.2f}s")
        return feedback_only

@app.local_entrypoint()
def main(input_text: str):
    model = CounselingFeedbackModel()
    result = model.generate.remote(input_text)
    print(result)

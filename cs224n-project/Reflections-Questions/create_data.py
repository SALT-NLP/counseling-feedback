import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from validate_skill import predict_skill
from datasets import load_dataset

dataset_id = "SALT-NLP/feedback_qesconv"
dataset = load_dataset(dataset_id, split="train")

# # Load dataset
# with open("./SALT-NLP-new/feedback_qesconv_full.json", "r", encoding="utf-8") as f:
#     dataset = json.load(f)
# Create a list to store the processed entries
binary_data = []

skills = ["questions", "reflections"]

for example in tqdm(dataset, desc="Processing entries", unit="entry"):
# for i in range(100, 110):
    # seeker_prompt = example['input'][-2].removeprefix("Seeker: ")
    # text = dataset[i]["text"]
    # print("index : ", i)
    text = example["text"]

    # Extract all helper responses
    lines = text.split("\n")
    # print("lines: ",lines)
    helper_responses = [line.replace("Helper: ", "").strip() for line in lines if line.startswith("Helper:")]
    original_response = "" if len(helper_responses) == 0 else helper_responses[-1]
    seeker_messages = [line.replace("Seeker: ", "").strip() for line in lines if line.startswith("Seeker:")]
    # print("seeker messages : ", seeker_messages)
    prompt = "" if len(seeker_messages) == 0 else seeker_messages[-1]
    # print("prompt : ", seeker_messages)
    # print("original response : ", original_response) 

    # # Extract feedback section
    response_section = text.split("### Response:")[-1].strip()
    response_JSON = json.loads(response_section) if response_section.startswith("{") else {}
    # print("response_JSON : ", response_JSON)
    
    alternative_response = response_JSON.get("alternative", "")  
    # print("alternative response : ", alternative_response)   

    # Process bad areas
    badareas = response_JSON.get("badareas", [])
    badareas = [skill.lower() for skill in badareas]  # Normalize case
    # print("badareas : ", badareas)

    # Create a binary entry
    binary_entry = {
        "Entry": text,
        "seeker-prompt": prompt,
        "last-helper-response": original_response,
        "alternative-response": alternative_response
    }
    for skill in skills:
      in_badarea = skill in badareas
      isQ = skill==skills[0]
      in_original = predict_skill(prompt, original_response, isQ)
      in_alternative = False if alternative_response == "" else predict_skill(prompt, alternative_response, isQ)

      badareas_shouldhave = 0
      badareas_shouldnothave = 0
      badareas_suboptimal = 0

      # Add to should_have if the skill is in badareas and is missing
      if skill in badareas:
        if in_original and in_alternative:
          badareas_suboptimal = 1
        if in_original and not in_alternative:
          badareas_shouldnothave = 1
        if not in_original and in_alternative:
          badareas_shouldhave = 1

      binary_entry[f"{skill}-badarea-shouldhave"] = badareas_shouldhave
      binary_entry[f"{skill}-badarea-optimal"] = badareas_suboptimal
      binary_entry[f"{skill}-badarea-shouldnothave"] = badareas_shouldnothave

    binary_data.append(binary_entry)

# Save as JSON
with open("augmented_dataset_reflections_questions_binary.json", "w", encoding="utf-8") as f:
    json.dump(binary_data, f, indent=4)

print(f"\nBinary dataset saved to augmented_dataset_binary.json")
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from setfit import SetFitModel
from has_skill import SkillDetector

# Load the model
model_path = "./finetune/best_setfit_deberta_.560"
model = SetFitModel.from_pretrained(model_path)

# Define the skill categories
skill_categories = [
    "empathy", "validation", "suggestion", "question",
    "professionalism", "self-disclosure", "structure"
]

# Load dataset
with open("./SALT-NLP-new/feedback_qesconv_full.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

skill_detector = SkillDetector(model)

# Create a list to store the processed entries
binary_data = []

for entry in tqdm(dataset, desc="Processing entries", unit="entry"):
    text = entry["text"]

    # Extract all helper responses
    lines = text.split("\n")
    helper_responses = [line.replace("Helper: ", "").strip() for line in lines if line.startswith("Helper:")]
    helper_text = " ".join(helper_responses)

    # Extract feedback section
    response_section = entry["text"].split("### Response:")[-1].strip()
    feedback = json.loads(response_section) if response_section.startswith("{") else {}

    # Process bad areas
    badareas = feedback.get("badareas", [])
    badareas = [skill.lower() for skill in badareas]  # Normalize case

    skill_predictions = skill_detector.detect_all_skills(helper_text)
    badareas_should_have = []
    badareas_suboptimal = []

    for skill in skill_categories:
        skill_present_original = skill_predictions[skill][0]

        # Add to should_have if the skill is in badareas and is missing
        if skill in badareas and not skill_present_original:
            badareas_should_have.append(skill)

        # Add to suboptimal if the skill is in badareas but was present
        if skill in badareas and skill_present_original:
            badareas_suboptimal.append(skill)

    # Create a binary entry
    binary_entry = {
        "Entry": text
    }

    for skill in skill_categories:
        binary_entry[f"{skill}-badarea-shouldhave"] = int(skill in badareas_should_have)
        binary_entry[f"{skill}-badarea-optimal"] = int(skill in badareas_suboptimal)

    binary_data.append(binary_entry)

# Save as JSON
with open("augmented_dataset_binary.json", "w", encoding="utf-8") as f:
    json.dump(binary_data, f, indent=4)

print(f"\nBinary dataset saved to augmented_dataset_binary.json")

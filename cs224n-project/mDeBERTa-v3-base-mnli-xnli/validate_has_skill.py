import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define skill categories (must match model training)
SKILL_CATEGORIES = [
    "Empathy", "Reflection", "Validation", "Suggestions", "Questions",
    "Professionalism", "Self-disclosure", "Structure"
]

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./finetune/fine_tuned_mDeBERTa"  # Adjust path if necessary

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode


def load_dataset(file_path: str):
    """Load JSON dataset"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_dataset(raw_dataset: list[dict]):
    """Extract text and labeled skills"""
    formatted_data = []
    for example in raw_dataset:
        text = example["text"]

        try:
            response_part = text.split("### Response:")[1].strip()
            response_json = json.loads(response_part)
        except (IndexError, json.JSONDecodeError):
            print(f"Skipping malformed entry: {text[:50]}")
            continue

        formatted_data.append({
            "text": text,
            "goodareas": response_json.get("goodareas", []),
            "badareas": response_json.get("badareas", []),
        })
    return formatted_data


# Model inference for a single skill
def predict_skill(text: str, skill: str, temperature: float = 1.5, percentile: float = 75):
    """Predict whether a specific skill is present in the text with temperature scaling."""
    
    if skill not in SKILL_CATEGORIES:
        raise ValueError(f"Invalid skill. Available skills: {SKILL_CATEGORIES}")

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits / temperature  # Apply temperature scaling
    probabilities = torch.sigmoid(logits).squeeze().tolist()  # Convert logits to probabilities

    # Convert scores into dictionary
    all_scores = {s: prob for s, prob in zip(SKILL_CATEGORIES, probabilities)}

    # Compute dynamic threshold based on percentile
    threshold = np.percentile(list(all_scores.values()), percentile)

    # Determine if the skill is present
    return (all_scores[skill] > threshold, round(all_scores[skill], 4))


# Model validation using `predict_skill()`
def validate_model(dataset, percentile=75, label_smoothing=0.1):
    """Evaluate model's skill prediction accuracy with label smoothing."""
    total_correct = 0
    total_predictions = 0
    total_actual = 0

    per_skill_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for example in tqdm(dataset, desc="Processing examples", unit="example"):
        text = example["text"]
        goodareas = set(example.get("goodareas", []))
        badareas = set(example.get("badareas", []))

        detected_skills = set()
        skill_scores = {}

        for skill in SKILL_CATEGORIES:
            is_present, normalized_score = predict_skill(text, skill, percentile=percentile)
            skill_scores[skill] = normalized_score  

            if is_present:
                detected_skills.add(skill)

            # track per-skill metrics with label smoothing
            if skill in goodareas and is_present:
                per_skill_metrics[skill]["TP"] += 1  
            elif skill in goodareas and not is_present:
                per_skill_metrics[skill]["FN"] += 1 - label_smoothing  
            elif skill in badareas and is_present:
                per_skill_metrics[skill]["FP"] += 1 - label_smoothing  

        print(f"TEXT: {text[:50]}...")
        print(f"Skill Predictions: {skill_scores}\n")

        total_correct += sum(1 for skill in goodareas if skill in detected_skills)
        total_predictions += len(detected_skills)
        total_actual += len(goodareas)

    precision = total_correct / total_predictions if total_predictions else 0
    recall = total_correct / total_actual if total_actual else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    per_skill_results = {}
    for skill, metrics in per_skill_metrics.items():
        tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
        precision_skill = tp / (tp + fp) if (tp + fp) else 0
        recall_skill = tp / (tp + fn) if (tp + fn) else 0
        f1_skill = (2 * precision_skill * recall_skill) / (precision_skill + recall_skill) if (precision_skill + recall_skill) else 0
        per_skill_results[skill] = {"Precision": precision_skill, "Recall": recall_skill, "F1-score": f1_skill}

    return {
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1-score": f1_score,
        "Per Skill Metrics": per_skill_results
    }


# -------------------- MAIN DRIVER CODE --------------------

if __name__ == "__main__":
    dataset = load_dataset("./test.json")
    formatted_dataset = format_dataset(dataset)

    # Run validation with the fine-tuned model
    results = validate_model(dataset=formatted_dataset, percentile=75)

    output_file = "skill_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\u2705 Validation results saved to {output_file}")

import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from PAIR_model import run_model
from cross_scorer_model import CrossScorerCrossEncoder
# from datasets import load_dataset, concatenate_datasets

def load_dataset(file_path: str):
    """Load JSON dataset"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def hasQuestion(text):
    return "?" in text

# Model inference for a single skill
def predict_skill(prompt: str, response: str, isQ: bool): #, threshold: float = 0.1):
    # score returned by PAIR = continuous between 0-1, 0: No reflection, 0.5: Simple reflection, 1: Complex reflection
    # score = run_model(prompt, response)
    # return score[0] > threshold
    if isQ:
        return hasQuestion(response)
    else: 
        return run_model(prompt, response) > 0.1

# Model validation using `predict_skill()`
def validate_model(dataset, threshold, label_smoothing):
    """Evaluate model's skill prediction accuracy with label smoothing."""

    metrics = {"TP": 0, "FP": 0, "FN": 0}
    correct = 0
    # count = 20
    # incorrect = []

    # for example in tqdm(dataset, desc="Processing examples", unit="example"):
    for example in dataset:
        seeker_prompt = example['input'][-2].removeprefix("Seeker: ")
        helper_response = example['input'][-1].removeprefix("Helper: ")
        alternative_response = example['annotations']['alternative']
        
        is_present_original = example['annotations']['original-hasquestion']
        is_present_alternative = example['annotations']['alternative-hasquestion']

        is_predicted = predict_skill(seeker_prompt, helper_response, threshold)

        # Track per-skill metrics with label smoothing
        if is_present_original == is_predicted: correct+=1
        # else: incorrect.append()
        if is_present_original and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_original and is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_original and not is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FP penalty

        is_predicted = predict_skill(seeker_prompt, alternative_response)

        if is_present_original == is_predicted: correct+=1
        if is_present_alternative and is_predicted:
            metrics["TP"] += 1  
        elif not is_present_alternative and is_predicted:
            metrics["FP"] += 1 - label_smoothing  # Reduce FN penalty
        elif is_present_alternative and not is_predicted:
            metrics["FN"] += 1 - label_smoothing  # Reduce FP penalty

        # print(f"TEXT: {text[:50]}...")
        # print(f"Skill Predictions: {skill_scores}\n")

    tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]
    precision_skill = tp / (tp + fp) if (tp + fp) else 0
    recall_skill = tp / (tp + fn) if (tp + fn) else 0
    f1_skill = (2 * precision_skill * recall_skill) / (precision_skill + recall_skill) if (precision_skill + recall_skill) else 0
    accuracy = correct / (len(dataset) *2)
    return {"Precision": precision_skill, "Recall": recall_skill, "Accuracy": accuracy, "F1-score": f1_skill}
    # return {"Accuracy": accuracy, "F1-score": f1_skill}


# -------------------- MAIN DRIVER CODE --------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = load_dataset("validation_set_skills.json")
    # print(dataset[0])

    results = validate_model(dataset, threshold = 0.1, label_smoothing=0.0)
    print(results)

    # Run validation with the fine-tuned model
    # for threshhold in [0.1, 0.6, 0.8]:
    #     for smoothing in np.arange(0, 1, 0.1):
    #         results = validate_model(dataset, threshhold, smoothing)
    #         print("Threshold: ", threshhold, " Smoothing: ", smoothing)
    #         print(results)

    # Save results
    # output_file = "reflection_validation_results.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2)

    # print(f"\u2705 Validation results saved to {output_file}")


    #not good
# Threshold:  0.1  Smoothing:  0.0
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3157894736842105)}
# Threshold:  0.1  Smoothing:  0.1
# {'Accuracy': 0.77, 'F1-score': np.float64(0.33898305084745767)}
# Threshold:  0.1  Smoothing:  0.2
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3658536585365853)}
# Threshold:  0.1  Smoothing:  0.30000000000000004
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3973509933774836)}
# Threshold:  0.1  Smoothing:  0.4
# {'Accuracy': 0.77, 'F1-score': np.float64(0.4347826086956523)}
# Threshold:  0.1  Smoothing:  0.5
# {'Accuracy': 0.77, 'F1-score': np.float64(0.48)}
# Threshold:  0.1  Smoothing:  0.6000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.5357142857142856)}
# Threshold:  0.1  Smoothing:  0.7000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.6060606060606062)}
# Threshold:  0.1  Smoothing:  0.8
# {'Accuracy': 0.77, 'F1-score': np.float64(0.697674418604651)}
# Threshold:  0.1  Smoothing:  0.9
# {'Accuracy': 0.77, 'F1-score': np.float64(0.8219178082191781)}
# Threshold:  0.6  Smoothing:  0.0
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3157894736842105)}
# Threshold:  0.6  Smoothing:  0.1
# {'Accuracy': 0.77, 'F1-score': np.float64(0.33898305084745767)}
# Threshold:  0.6  Smoothing:  0.2
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3658536585365853)}
# Threshold:  0.6  Smoothing:  0.30000000000000004
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3973509933774836)}
# Threshold:  0.6  Smoothing:  0.4
# {'Accuracy': 0.77, 'F1-score': np.float64(0.4347826086956523)}
# Threshold:  0.6  Smoothing:  0.5
# {'Accuracy': 0.77, 'F1-score': np.float64(0.48)}
# Threshold:  0.6  Smoothing:  0.6000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.5357142857142856)}
# Threshold:  0.6  Smoothing:  0.7000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.6060606060606062)}
# Threshold:  0.6  Smoothing:  0.8
# {'Accuracy': 0.77, 'F1-score': np.float64(0.697674418604651)}
# Threshold:  0.6  Smoothing:  0.9
# {'Accuracy': 0.77, 'F1-score': np.float64(0.8219178082191781)}
# Threshold:  0.8  Smoothing:  0.0
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3157894736842105)}
# Threshold:  0.8  Smoothing:  0.1
# {'Accuracy': 0.77, 'F1-score': np.float64(0.33898305084745767)}
# Threshold:  0.8  Smoothing:  0.2
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3658536585365853)}
# Threshold:  0.8  Smoothing:  0.30000000000000004
# {'Accuracy': 0.77, 'F1-score': np.float64(0.3973509933774836)}
# Threshold:  0.8  Smoothing:  0.4
# {'Accuracy': 0.77, 'F1-score': np.float64(0.4347826086956523)}
# Threshold:  0.8  Smoothing:  0.5
# {'Accuracy': 0.77, 'F1-score': np.float64(0.48)}
# Threshold:  0.8  Smoothing:  0.6000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.5357142857142856)}
# Threshold:  0.8  Smoothing:  0.7000000000000001
# {'Accuracy': 0.77, 'F1-score': np.float64(0.6060606060606062)}
# Threshold:  0.8  Smoothing:  0.8
# {'Accuracy': 0.77, 'F1-score': np.float64(0.697674418604651)}
# Threshold:  0.8  Smoothing:  0.9
# {'Accuracy': 0.77, 'F1-score': np.float64(0.8219178082191781)}
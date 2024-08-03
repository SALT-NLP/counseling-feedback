import json
import numpy as np
import csv
from datasets import load_dataset
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
from evaluate import load
from tqdm import tqdm

# Thresholds (95th percentile)
BLEU_THRESHOLD = 0.1249
ROUGE1_THRESHOLD = 0.4898
ROUGE2_THRESHOLD = 0.2612
ROUGEL_THRESHOLD = 0.4182
BERTSCORE_THRESHOLD = 0.8407

# Load the dataset
dataset = load_dataset("SALT-NLP/feedback_qesconv")
train_data = dataset['train']

# Initialize scorers
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bertscore = load("bertscore")

# Initialize score lists and nitpicky alternatives list
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
bert_scores = []
nitpicky_alternatives = []

for idx, item in enumerate(tqdm(train_data)):
    # Extract the feedback JSON
    feedback_json = json.loads(item['text'].split('Response:')[-1])
    
    # Only process items where perfect is false
    if not feedback_json['perfect']:
        # Extract the last helper response and the alternative
        conversation = item['text'].split('Helper:')
        last_helper_response_and_feedback = conversation[-1].split('Seeker:')[0].strip()
        last_helper_response = last_helper_response_and_feedback.split('### Response:')[0]
        
        if idx % 100 == 0:
            print("conversation: ", conversation)
            print("last_helper_response_and_feedback: ", last_helper_response_and_feedback)
            print("last_helper_response: ", last_helper_response)
        alternative = feedback_json['alternative']

        # Compute BLEU score
        bleu = bleu_score.sentence_bleu([last_helper_response.split()], alternative.split())
        bleu_scores.append(bleu)

        # Compute ROUGE scores
        rouge_raw = scorer.score(alternative, last_helper_response)
        rouge1 = rouge_raw['rouge1'].fmeasure
        rouge2 = rouge_raw['rouge2'].fmeasure
        rougeL = rouge_raw['rougeL'].fmeasure
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougeL_scores.append(rougeL)

        # Compute BERTScore
        bert_score = bertscore.compute(predictions=[alternative],
                                       references=[last_helper_response], 
                                       model_type="distilbert-base-uncased")
        bert = bert_score['f1'][0]
        bert_scores.append(bert)

        # Check if this item exceeds all thresholds
        if (bleu > BLEU_THRESHOLD and
            rouge1 > ROUGE1_THRESHOLD and
            rouge2 > ROUGE2_THRESHOLD and
            rougeL > ROUGEL_THRESHOLD and
            bert > BERTSCORE_THRESHOLD):
            nitpicky_alternatives.append({
                'index': idx,
                'bleu': bleu,
                'rouge1': rouge1,
                'rouge2': rouge2,
                'rougeL': rougeL,
                'bert': bert,
                'last_response': last_helper_response,
                'alternative': alternative
            })

def print_stats(name, scores):
    percentiles = np.percentile(scores, [0, 25, 50, 75, 85, 90, 95, 100])
    print(f'{name}:')
    print(f'  Min (0th): {percentiles[0]:.4f}')
    print(f'  25th Percentile: {percentiles[1]:.4f}')
    print(f'  Median (50th): {percentiles[2]:.4f}')
    print(f'  75th Percentile: {percentiles[3]:.4f}')
    print(f'  85th Percentile: {percentiles[4]:.4f}')
    print(f'  90th Percentile: {percentiles[5]:.4f}')
    print(f'  95th Percentile: {percentiles[6]:.4f}')
    print(f'  Max (100th): {percentiles[7]:.4f}')
    print(f'  Mean: {np.mean(scores):.4f}')

# Print results
print(f'Number of imperfect responses evaluated: {len(bleu_scores)}')
print_stats('BLEU', bleu_scores)
print_stats('ROUGE-1', rouge1_scores)
print_stats('ROUGE-2', rouge2_scores)
print_stats('ROUGE-L', rougeL_scores)
print_stats('BERTScore', bert_scores)

# Save nitpicky alternatives to CSV
csv_filename = 'nitpicky_alternatives.csv'
csv_fields = ['index', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'bert', 'last_response', 'alternative']

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()
    for item in nitpicky_alternatives:
        writer.writerow(item)

print(f"\nNumber of nitpicky alternatives: {len(nitpicky_alternatives)}")
print(f"Nitpicky alternatives have been saved to {csv_filename}")

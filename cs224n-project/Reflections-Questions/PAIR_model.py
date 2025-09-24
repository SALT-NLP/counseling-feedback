# Copied from https://github.com/MichiganNLP/PAIR/blob/main/
from transformers import AutoTokenizer, AutoModel
import torch
from cross_scorer_model import CrossScorerCrossEncoder

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Load pre-trained weights")
scorer_path = "Weights/reflection_scorer_weight.pt"  # Your weight file
c_ckpt = torch.load(scorer_path, weights_only=True,map_location=torch.device(device))

# print(c_ckpt.keys())

print("Load tokenizer and model")
model_name = "roberta-base"
encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cross_scorer = CrossScorerCrossEncoder(encoder).to(device)

# print("cross_scorer expected layer names: ", cross_scorer.state_dict().keys())

def checkPercentOverlap():
    ckpt_keys = set(c_ckpt.keys())  # Keys from loaded checkpoint
    model_keys = set(cross_scorer.state_dict().keys())  # Expected keys in the model

    # Find the number of matching keys
    matching_keys = ckpt_keys.intersection(model_keys)
    num_matching = len(matching_keys)
    total_keys = len(model_keys)  # Use model keys as the reference

    # Compute the match percentage
    match_percentage = (num_matching / total_keys) * 100 if total_keys > 0 else 0

    # Print results
    print(f"Matching Keys: {num_matching}/{total_keys} ({match_percentage:.2f}%)")
    print(f"Missing Keys in Checkpoint: {model_keys - ckpt_keys}")
    print(f"Extra Keys in Checkpoint: {ckpt_keys - model_keys}")

# checkPercentOverlap()
# Matching Keys: 201/201 (100.00%)
# Missing Keys in Checkpoint: set()
# Extra Keys in Checkpoint: {'cross_encoder.embeddings.position_ids'}

cross_scorer.load_state_dict(c_ckpt, strict=False) #["model_state_dict"])
cross_scorer.eval()  # Set model to evaluation mode

# Function to run inference
def run_model(prompt, response):
    # print("prompt", prompt)
    # print("resp", response)
    model=cross_scorer
    batch = tokenizer(prompt, response, padding='longest', truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        sims = model.score_forward(**batch).sigmoid().flatten().tolist()
    return sims[0]

# print("False" * 30)
# #should be 0
# prompt = "Hi. I don't know... sort of down. Sort of empty... How about you?"
# response = "I'm sorry to hear that you're feeling this way. Can you tell me a bit more about what you're going through?"
# # input("Give prompt: ")
# score = run_model(cross_scorer, tokenizer, prompt, response)
# print("Score: ", score) # Score:  [0.00032032481976784766]
# print()

# print("True" * 30)
# # should be 1
# prompt = "What if they're disappointed"
# response = "Dealing with the fear of disappointing the ones we love can be really tough. Can you tell me more about what you're afraid might happen if you tell them?"
# # input("Give prompt: ")
# score = run_model(cross_scorer, tokenizer, prompt, response)
# print("Score: ", score) #Score:  [0.0004518316709436476]
# print()

# print("True" * 30)
# # should be 1
# prompt = "Is that ghosting?"
# response = "It seems like you're questioning whether this could be considered as 'ghosting'. What are your thoughts and feelings about the situation if it is, in fact, ghosting?"
# # input("Give prompt: ")
# score = run_model(cross_scorer, tokenizer, prompt, response) 
# print("Score: ", score) #Score:  [0.9962798953056335]
# print()


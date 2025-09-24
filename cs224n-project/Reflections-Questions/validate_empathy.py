import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import AutoTokenizer

# https://huggingface.co/MoaazZaki/bert-empathy
# tokenizer = AutoTokenizer.from_pretrained("MoaazZaki/bert-empathy")
# model = AutoModelForSequenceClassification.from_pretrained("MoaazZaki/bert-empathy")

# https://huggingface.co/paragon-analytics/bert_empathy
# tokenizer = AutoTokenizer.from_pretrained("paragon-analytics/bert_empathy")
# model = AutoModelForSequenceClassification.from_pretrained("paragon-analytics/bert_empathy")

# https://huggingface.co/mmillet/distilrubert-tiny-cased-conversational-v1_finetuned_empathy_classifier
tokenizer = AutoTokenizer.from_pretrained("mmillet/distilrubert-tiny-cased-conversational-v1_finetuned_empathy_classifier")
model = AutoModelForSequenceClassification.from_pretrained("mmillet/distilrubert-tiny-cased-conversational-v1_finetuned_empathy_classifier")

model.eval()

def classify(x):
    encoded_input = tokenizer(x, return_tensors='pt').to(device)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = tf.nn.softmax(scores)
    return scores.numpy()[1]

def classify_text(text):
    print("Number of classes:", model.config.num_labels)

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get raw logits (scores before softmax)
    logits = outputs.logits
    print("Logits:", logits)

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get predicted class and confidence score
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    return predicted_class, confidence

def score_empathy(text):
    prompt = f"Evaluate the following statement on empathy, giving a score from 0 to 1:\n\n'{text}'\n\nEmpathy Score:"
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    score = float(response.split()[-1])  # Adjust parsing if needed
    return score

def test_model():
    responses = ["I can understand how that feels - I have not been able to go out for weeks either. Are you able to have visitors?", #T
                 "I'm really sorry that you're feeling this way. How have these feelings been affecting your daily life?", #T
                 "have you thought about scheduling free time for your self ? It can give you something to look forward to and help keep you motivated /Almost like a reward for finishing your school work", #F
                 "Sounds like you're rushing to learn the material right before class. That can be stressful. What are some strategies you've tried to manage your time better? How effective were they?", #T
                 "It might be worthwhile to recommend him look into getting some counseling about his issues that you seem to think are a real issue.", #F
                 "How does it feel being back home? Do you find any comfort in it?", #F    
                 "Bob" #F                
        ]
    for response in responses:
      result = score_empathy(response) 
      print(result) 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    test_model()
    # print(classify_text("bob"))
    
# distilrubert results:
# (1, 0.4327591359615326)
# (1, 0.5065935254096985)
# (1, 0.43606334924697876)
# (1, 0.6738070249557495)
# (1, 0.49049681425094604)
# (1, 0.7863013744354248)
# (1, 0.8711757659912109)

# roberta results:
# 0.8613588
# 0.91284806
# 0.93154764
# 0.8176627
# 0.83720064
# 0.88704276
# 0.88943297
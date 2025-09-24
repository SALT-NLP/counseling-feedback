import torch
import numpy as np
from setfit import SetFitModel
from sentence_transformers import SentenceTransformer, util

class SkillDetector:
    def __init__(self, model, threshold=0.4):
        """
        Initializes the skill detector with a fine-tuned SetFit model.
        Uses skill descriptions for better decision-making.
        """
        self.model = model
        # use a light weight sentence encoder
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
        self.threshold = threshold  # Probability threshold for classification

        # define skills (excluding "Reflection")
        self.SKILL_CATEGORIES = [
            "empathy", "validation", "suggestion", "question",
            "professionalism", "self-disclosure", "structure"
        ]


        # define descriptions to provide context
        self.skill_descriptions = {
            "empathy": "Demonstrates understanding and emotional connection by acknowledging the seeker's emotions.",
            "validation": "Expresses agreement or acknowledgment that the other person's feelings or opinions are reasonable.",
            "suggestion": "Provides actionable advice or alternative solutions to help the other person navigate their situation.",
            "question": "Asks open-ended questions to encourage deeper conversation and engagement. If a '?' is present in the text, it's likely a question.",
            "professionalism": "Maintains a formal, respectful, and non-judgmental tone.",
            "self-disclosure": "Shares personal experiences to relate to the seeker while maintaining professional boundaries.",
            "structure": "Clearly organizes responses into logical, well-formed sentences with a beginning, middle, and end."
        }

        # precompute skill description embeddings
        self.skill_embeddings = {
            skill: self.embedding_model.encode(desc, convert_to_tensor=True)
            for skill, desc in self.skill_descriptions.items()
        }

    def has_skill(self, text: str, skill: str) -> tuple[bool, float]:
        """
        Predicts whether a given skill is present in the input text.
        Uses both the fine-tuned SetFit model and skill descriptions.
        """
        if skill not in self.SKILL_CATEGORIES:
            raise ValueError(f"Invalid skill. Available skills: {self.SKILL_CATEGORIES}")

        # run model prediction
        prediction = self.model.predict([text])  
        prediction = prediction.item() if isinstance(prediction, torch.Tensor) else float(prediction)

        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        description_embedding = self.skill_embeddings[skill]
        similarity_score = util.pytorch_cos_sim(text_embedding, description_embedding).item()

        # normalize
        mean_sim = np.mean([tensor.cpu().numpy() for tensor in self.skill_embeddings.values()])
        std_sim = np.std([tensor.cpu().numpy() for tensor in self.skill_embeddings.values()])
        normalized_score = (similarity_score - mean_sim) / (std_sim + 1e-8)  

        
        confidence = (prediction + normalized_score) / 2  

        skill_present = confidence >= self.threshold

        return skill_present, confidence  # (True/False, Confidence Score)

    def detect_all_skills(self, text):
        """
        Detects all skills in a given text using the fine-tuned model and skill descriptions.
        Uses `has_skill()` for better decision-making.
        Returns a dictionary with skill presence (True/False) and confidence scores.
        """
        skill_results = {}
        for skill in self.SKILL_CATEGORIES:
            skill_present, confidence = self.has_skill(text, skill) 
            skill_results[skill] = (skill_present, confidence)  

        return skill_results  


"""
###### Example Usage ########

detector = SkillDetector()

text = "I understand how you're feeling, it's tough going through this."
skill = "Empathy"

# Test single skill
result = detector.has_skill(text, skill)
print(f"Skill '{skill}' present: {result[0]}, Confidence Score: {result[1]:.4f}")

# Test all skills
all_skills = detector.detect_all_skills(text)
print("All Skill Predictions:", all_skills)
"""

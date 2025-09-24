import numpy as np
from sentence_transformers import SentenceTransformer, util

"""
file description:
- this file uses model sentence-t5-large which didn't perform the best, so moving onto finetuning model for better results.

"""
class SkillDetector:
    def __init__(self, model_name="sentence-t5-large"):
        self.model = SentenceTransformer(model_name)
        self.skill_descriptions = {
            "Empathy": "Demonstrates understanding and emotional connection by acknowledging the seeker's emotions.",
            "Reflection": "Paraphrases or repeats key parts of what the other person is saying to confirm understanding.",
            "Validation": "Expresses agreement or acknowledgment that the other person's feelings or opinions are reasonable.",
            "Suggestions": "Provides actionable advice or alternative solutions to help the other person navigate their situation.",
            "Questions": "Asks open-ended questions to encourage deeper conversation and engagement." + "if a ? is in the text its True",
            "Professionalism": "Maintains a formal, respectful, and non-judgmental tone.",
            "Self-disclosure": "Shares personal experiences to relate to the seeker while maintaining professional boundaries.",
            "Structure": "Clearly organizes responses into logical, well-formed sentences with a beginning, middle, and end."
        }

        self.trait_embeddings = {
            skill: self.model.encode(desc, convert_to_tensor=True)
            for skill, desc in self.skill_descriptions.items()
        }

    def has_skill(self, text: str, skill: str, threshold: float) -> tuple[bool, float]:
        """
        checks if a given skill is present in the input text using semantic similarity.
        """
        if skill not in self.skill_descriptions:
            raise ValueError(f"invalid skill. available skills: {list(self.skill_descriptions.keys())}")

        text_embedding = self.model.encode(text, convert_to_tensor=True)
        all_scores = {
            s: util.pytorch_cos_sim(text_embedding, self.trait_embeddings[s]).item()
            for s in self.skill_descriptions.keys()
        }

        # compute mean and standard deviation of scores
        mean_score = np.mean(list(all_scores.values()))
        std_dev = np.std(list(all_scores.values()))

        # apply z-score normalization
        normalized_score = (all_scores[skill] - mean_score) / (std_dev + 1e-8)  

        return (normalized_score > threshold, normalized_score)


"""
###### example usage ########

    detector = SkillDetector()

    text = "The goal is to provide a structured response."
    skill = "Structure"

    result = detector.has_skill(text, skill)
    print(f"Skill '{skill}' present: {result}")

"""

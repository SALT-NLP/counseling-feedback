import json
from datasets import Dataset

test_data_file = "model_training/data/feedback_qesconv/train.json"

SKILL_OPTIONS = [
    "Reflections", "Validation", "Empathy", "Questions",
    "Suggestions", "Self-disclosure", "Structure", "Professionalism"
]

def read_data(data_file):
    """Reads JSON data from a file."""
    with open(data_file, "r") as f:
        return json.load(f)

def parse_data_point(data_point, is_test_data=False):
    """Parses a single data point from the dataset."""
    data_text = data_point['text']
    keys = data_text.split('###')

    # Extract the instruction, input, and response
    instruction = keys[1].strip()
    input_text = keys[2].strip()
    response_text = keys[3].strip()

    # Extract the conversation and helper indices
    conv_index = data_point['conv_index']
    helper_index = data_point['helper_index']

    # Format the input to remove the first line
    input_lines = input_text.split('\n')
    input_lines = input_lines[1:]
    input_formatted = [line.strip() for line in input_lines if line.strip()]  # Remove empty lines and extra spaces

    # Don't parse response data since test data is empty
    if is_test_data:
        response_data = ""
    else:
        response_data = json.loads(response_text[response_text.index(":") + 1:].strip())

    return {
        'conv_index': conv_index,
        'helper_index': helper_index,
        'instruction': instruction,
        'input': input_formatted,
        'response': response_data
    }

def classification_labels_from_feedback(feedback, skill_options):
    """
    reflections-goodareas: 0 or 1
    validation-goodareas: 0 or 1
    empathy-goodareas: 0 or 1
    ...
    reflections-badareas: 0 or 1
    validation-badareas: 0 or 1 
    """
    print("Feedback:\n", feedback)
    assert isinstance(feedback, dict)
    labels = {}
    if 'goodareas' in feedback:
        labels.update({f"{skill}-goodareas": 1 if skill in feedback['goodareas'] else 0 for skill in skill_options})
    else:
        labels.update({f"{skill}-goodareas": 0 for skill in skill_options})
    if 'badareas' in feedback:
        labels.update({f"{skill}-badareas": 1 if skill in feedback['badareas'] else 0 for skill in skill_options})
    else:
        labels.update({f"{skill}-badareas": 0 for skill in skill_options})
    return labels

def restructure_data_for_classification(data_points):
    restructured_datapoints = []
    for data_point in data_points:
        feedback = data_point['response']
        class_labels = classification_labels_from_feedback(feedback, SKILL_OPTIONS)
        restructured_data = {
            'conv_index': data_point['conv_index'],
            'helper_index': data_point['helper_index'],
            'input': data_point['input'],
        }
        restructured_data.update(class_labels)
        restructured_datapoints.append(restructured_data)
    return restructured_datapoints

def main():

    data = read_data(test_data_file)
    data_points = [parse_data_point(data_point) for data_point in data]
    print(f"Num datapoints: {len(data_points)}")
    print("Data points:\n", data_points[:5])
    classification_data = restructure_data_for_classification(data_points)
    print("Classification data:\n", classification_data[:5])
    # Convert to Dataset format and push to hub
    
    # Convert to Dataset
    dataset = Dataset.from_list(classification_data)
    
    # Push to the Hugging Face Hub - replace with your desired dataset name
    # Note: You need to be logged in via huggingface-cli login first
    # dataset.push_to_hub("youralien/feedback_qesconv_16wayclassification")
if __name__ == "__main__":
    main()
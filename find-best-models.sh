#!/bin/bash

# Script to find the highest eval_f1 score in each model directory

# Directory pattern to search for
# MODEL_PREFIX="roberta-Validation-goodareas-eval_FeedbackESConv5pp_CARE10pp-sweeps-best-wdbkc6pj-*"
MODEL_PREFIX="roberta-Self-disclosure-badareas-eval_FeedbackESConv5pp_CARE10pp-sweeps-best-fk58yziy-*"

# Function to extract the highest eval_f1 score from a trainer_state.json file
function extract_highest_eval_f1() {
    local file=$1
    local model_dir=$2
    
    # Use jq to parse the JSON and extract all eval_f1 scores
    # Then find the maximum value using sort and tail
    if command -v jq &> /dev/null; then
        # If jq is available, use it for proper JSON parsing
        highest_f1=$(jq -r '.log_history[] | select(.eval_f1 != null) | .eval_f1' "$file" | sort -n | tail -1)
        epoch=$(jq -r '.log_history[] | select(.eval_f1 == '"$highest_f1"') | .epoch' "$file" | head -1)
        step=$(jq -r '.log_history[] | select(.eval_f1 == '"$highest_f1"') | .step' "$file" | head -1)
        
        echo "$model_dir,$highest_f1,$epoch,$step"
    else
        # Fallback to grep and awk if jq is not available (less reliable)
        echo "Warning: jq not found. Using fallback method for $file" >&2
        grep -o '"eval_f1": [0-9.]*' "$file" | awk -F: '{print $2}' | sort -n | tail -1 | tr -d ' ' | xargs -I{} echo "$model_dir,{},NA,NA"
    fi
}

# Main script
echo "Model Directory,Highest eval_f1,Epoch,Step"

# Find all model directories matching the pattern
for model_dir in $MODEL_PREFIX; do
    # Skip if not a directory
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    # Find all trainer_state.json files in checkpoint directories
    find "$model_dir" -name "trainer_state.json" | while read -r json_file; do
        # Extract and display the highest eval_f1 score
        extract_highest_eval_f1 "$json_file" "$model_dir"
    done
done > temp_results.csv

# Make sure we have a header line
if [[ $(wc -l < temp_results.csv) -gt 0 ]]; then
  header=$(head -n 1 temp_results.csv)
  
  # Sort the data rows using proper numeric sort for the second field (eval_f1)
  # LC_ALL=C forces consistent sorting behavior across different environments
  (echo "$header" && tail -n +2 temp_results.csv | LC_ALL=C sort -t, -k2 -gr) > results.csv
else
  # If no results were found, just create an empty file with the header
  echo "Model Directory,Highest eval_f1,Epoch,Step" > results.csv
fi

rm temp_results.csv

# Display all results
cat results.csv

# Extract and display the model with the highest eval_f1 score
echo -e "\n===== MODEL WITH HIGHEST EVAL_F1 SCORE ====="
head -n 1 results.csv  # Display header
head -n 2 results.csv | tail -n 1  # Display top result

import os
import json
import argparse
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def save_split(data, output_dir, split_name):
   os.makedirs(output_dir, exist_ok=True)
   file_path = os.path.join(output_dir, f"{split_name}.json")
   with open(file_path, 'w') as f:
       json.dump([{
           "text": d["text"],
           "helper_index": d["helper_index"],
           "conv_index": d["conv_index"]
       } for d in data], f, indent=4)
   print(f"Created {split_name} split with {len(data)} examples in {file_path}")


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                       help="Specify which split to generate (train, test, or both)")
   parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory where the output files will be saved")
   args = parser.parse_args()

   ds = load_dataset("avylor/feedback_qesconv")
   data = list(ds['train'])
  
   # create train/test split
   train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
  
   # save the specified splits
   if args.split in ["train", "both"]:
       save_split(train_data, args.output_dir, "train")
   if args.split in ["test", "both"]:
       save_split(test_data, args.output_dir, "test")


if __name__ == "__main__":
   main()




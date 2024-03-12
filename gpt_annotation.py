import json
import time
from openai import OpenAI
from tqdm import tqdm
import argparse

client = OpenAI()
MAX_TRIES = 3
TEMPERATURE = 0.8

# send as system message, instructions
GPT_INSTRUCTION_FILE = 'prompts/prompt_gpt_instruction.txt'

# send as user message, exemplars
GPT_INPUT_PREFIX_FILE = 'prompts/prompt_gpt_input_prefix.txt'


def annotate_with_gpt(gpt_prompt_input_file, model='gpt-3.5-turbo'):

    with open(GPT_INSTRUCTION_FILE, 'r') as file:
        prompt_instruction = file.read()
    with open(GPT_INPUT_PREFIX_FILE, 'r') as file:
        gpt_input_prefix = file.read()

    with open(gpt_prompt_input_file) as json_file:
        gpt_prompt_inputs = json.load(json_file)

    finetuning_responses = []
    stats = []
    skipped = []

    for prompt_input in tqdm(gpt_prompt_inputs, desc=f'Annotating with {model}'):
        formatted_input = f'{gpt_input_prefix}**Conversation**\n\n{prompt_input["input"]}\n\n**Annotation**\n\n'

        for t in range(MAX_TRIES):
            try:
                print(f'Attempt #{t}')

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt_instruction},
                              {"role": "user", "content": formatted_input}],
                    temperature=TEMPERATURE)

                response_json = json.loads(response.choices[0].message.content)
                annotations = response_json["annotations"]

                # check if all fields present
                for i, helper_index in enumerate(prompt_input["helper_indices"]):
                    ann = annotations[i]
                    if "helper" not in ann:
                        raise Exception("No helper in annotation!")
                    if "goodareas" not in ann and "areas" not in ann:
                        raise Exception("No goodareas or areas in annotation!")
                    if "perfect" not in ann:
                        raise Exception("No perfect in annotation!")
                    if ann["perfect"] == False:
                        if "feedback" not in ann:
                            raise Exception("No feedback in annotation!")
                        if "badareas" not in ann:
                            raise Exception("No areas in annotation!")
                        if "alternative" not in ann:
                            raise Exception("No alternative in annotation!")

                for i, helper_index in enumerate(prompt_input["helper_indices"]):
                    # we annotate in chunks of 5 helper's responses, we skip the first two if this is
                    # not the conversation start
                    if prompt_input["helper_indices"][0] != 0 and i < 2:
                        continue
                    finetuning_responses.append({"output": annotations[i]})
                    finetuning_responses[-1]['helper_index'] = helper_index
                    finetuning_responses[-1]['gpt_prompt_input'] = formatted_input
                    finetuning_responses[-1]['conv_index'] = prompt_input["conv_index"]

                stats.append(dict(response.usage))
                print(stats[-1])

                break
            except Exception as e:
                print(f'Failed to get/parse GPT output:', e)
                time.sleep(2)

                # edge case handling, skip the prompt
                if t == MAX_TRIES-1:
                    if prompt_input["helper_indices"][0] == 0:
                        for helper_index in prompt_input["helper_indices"]:
                            skipped.append((prompt_input["conv_index"], helper_index))
                    else:
                        for helper_index in prompt_input["helper_indices"][2:]:
                            skipped.append((prompt_input["conv_index"], helper_index))

    completion_tokens_total = 0
    prompt_tokens_total = 0
    for stat in stats:
        completion_tokens_total += stat["completion_tokens"]
        prompt_tokens_total += stat["prompt_tokens"]

    stats_global = {"completion_tokens_total": completion_tokens_total, "prompt_tokens_total": prompt_tokens_total,
                    "annotation_success_rate": len(stats) / len(gpt_prompt_inputs)}

    print(f'Completion tokens total: {completion_tokens_total}')
    print(f'Prompt tokens total: {prompt_tokens_total}')
    print(f'Annotation success rate: {len(stats) / len(gpt_prompt_inputs)}')
    print(f'--------------Finished annotating with {model}------------------')

    # save responses
    # with open(f'created_datasets/finetuning_responses_{model}_{TEMPERATURE}.json', 'w') as outfile:
    #     json.dump(finetuning_responses, outfile)

    return finetuning_responses, stats, stats_global, skipped


def generate_finetuning_dataset(finetuning_responses, stats, stats_global, skipped, finetuning_prompt_input_file, model):

    with open(finetuning_prompt_input_file) as json_file:
        finetuning_inputs = json.load(json_file)

    suffix = finetuning_prompt_input_file.split('_')[-1][:-5]

    dataset_for_finetuning = []
    dataset_full_data = []

    input_pointer = 0

    for response_pointer, finetuning_response in enumerate(finetuning_responses):


        # omit skipped in annotation examples
        while (input_pointer < len(finetuning_inputs) and
               (finetuning_inputs[input_pointer]['conv_index'],
                finetuning_inputs[input_pointer]['helper_index']) in skipped):
            input_pointer += 1

        finetuning_input = finetuning_inputs[input_pointer]

        new_entry = {}
        new_entry['instruction'] = "Give feedback to the Helper's last response."
        new_entry['input'] = finetuning_input['input']
        new_entry['output'] = finetuning_response['output']
        dataset_for_finetuning.append(new_entry)

        dataset_full_data.append(new_entry.copy())
        dataset_full_data[-1]['helper_index'] = finetuning_response['helper_index']
        dataset_full_data[-1]['gpt_prompt_input'] = finetuning_response['gpt_prompt_input']
        dataset_full_data[-1]['conv_index'] = finetuning_response['conv_index']

        ### sanity checks
        assert finetuning_input['helper_index'] == finetuning_response['helper_index']
        assert finetuning_input['conv_index'] == finetuning_response['conv_index']

        input_pointer += 1

    # save the dataset for finetuning as json file
    with open(f'created_datasets/dataset_for_finetuning_{model}_{TEMPERATURE}_{suffix}.json', 'w') as outfile:
        json.dump(dataset_for_finetuning, outfile)
    # save full dataest as json file
    with open(f'created_datasets/dataset_full_data_{model}_{TEMPERATURE}_{suffix}.json', 'w') as outfile:
        json.dump(dataset_full_data, outfile)
    # save stats
    with open(f'created_datasets/stats_{model}_{TEMPERATURE}_{suffix}.json', 'w') as outfile:
        json.dump(stats, outfile)
    # save global stats
    with open(f'created_datasets/stats_global_{model}_{TEMPERATURE}_{suffix}.json', 'w') as outfile:
        json.dump(stats_global, outfile)
    # save skipped prompts
    with open(f'created_datasets/skipped_prompts_{model}_{TEMPERATURE}_{suffix}.json', 'w') as outfile:
        json.dump(skipped, outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpt_prompt_input_file', type=str, help='The GPT prompt input file')
    parser.add_argument('--finetuning_prompt_input_file', type=str, help='The finetuning prompt input file')
    parser.add_argument('--model', type=str, help='The model string')

    args = parser.parse_args()

    # Print the arguments
    print("GPT Prompt Input File:", args.gpt_prompt_input_file)
    print("Finetuning Prompt Input File:", args.finetuning_prompt_input_file)
    print("Model:", args.model)

    finetuning_responses, stats, stats_global, skipped = annotate_with_gpt(args.gpt_prompt_input_file, model=args.model)
    generate_finetuning_dataset(finetuning_responses=finetuning_responses, stats=stats, stats_global=stats_global,
                                skipped=skipped, finetuning_prompt_input_file=args.finetuning_prompt_input_file,
                                model=args.model)


    # please note that the order needs to match between those two
    # gpt_prompt_input_file = 'prompts/prompts_input_part_train_GPT_40-49.json'
    # finetuning_prompt_input_file = 'prompts/prompts_input_part_train_40-49.json'
    # model = 'gpt-4'
    # pre gpt-4 annotation data was preprocesssed using utils.py preprocess_dataset function
    # for i in range(0, 400, 10):
    #   preprocess_dataset('train', gpt=True, save_suffix=f"_{i}-{i + 9}", indices_range=[i for i in range(i, i + 10)])
    #   preprocess_dataset('train', gpt=False, save_suffix=f"_{i}-{i + 9}",indices_range=[i for i in range(i, i + 10)])

    # example usage:
    # poetry run python gpt_annotation.py --gpt_prompt_input_file prompts/prompts_input_part_train_GPT_0-9.json --finetuning_prompt_input_file prompts/prompts_input_part_train_0-9.json --model gpt-4

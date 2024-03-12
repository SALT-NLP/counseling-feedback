import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os

from C99 import C99

DATASET = "qesconv"
# DATASET = "qesconvdpo" # set for dpo data processing

# ------------------------------------- DATASET ESCONV UTILS ----------------------------------------------- #

def load_esconv_dataset():
    if DATASET == "qesconv":
        with open('data/q-esconv.json') as json_file:
            data = json.load(json_file)
            dataset = {'train': data[:400], 'test': data[400:]}
    elif DATASET == "qesconvdpo":
        with open('data/q-esconv-dpo.json') as json_file:
            data = json.load(json_file)
            dataset = {'train': data, 'test': []}
    return dataset

# ------------------------------------- SEGMENTATION UTILS ----------------------------------------------- #
def conversation_to_embeddings(emb_model, conv):
    """Returns the embeddings of a conversation."""
    embeddings = emb_model.encode(conv)
    return embeddings


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def process_datapoint(datapoint):
    """Returns the conversation and speaker of a datapoint dataset['train'][ind]"""
    dialog = json.loads(datapoint['text'])['dialog']
    conv = [dialog[line]['text'] for line in range(len(dialog))]
    speaker = [dialog[line]['speaker'] for line in range(len(dialog))]
    speaker = [{'usr': 'Seeker', 'sys': 'Helper'}[speaker[i]] for i in range(len(speaker))]

    # combine utterances from the same speaker
    conv_new = [conv[0]]
    speaker_new = [speaker[0]]
    for i in range(1, len(conv)):
        if speaker[i] == speaker[i-1]:
            if conv_new[-1][-1] == '.' or conv_new[-1][-1] == '?':
                conv_new[-1] += ' ' + conv[i]
            else:
                conv_new[-1] += '. ' + conv[i]
        else:
            conv_new.append(conv[i])
            speaker_new.append(speaker[i])

    return conv_new, speaker_new


def create_prompt_inputs(emb_model, seg_model, conv_index, conv, speaker, nr_segments_to_include=2):
    """Returns a list of prompt inputs and a list of their lengths [nr tokens] for a conversation.
    It segments the conversation, the for each helpers's utterance adds the context of previous segment/s"""
    embeddings = conversation_to_embeddings(emb_model=emb_model, conv=conv)
    segmentation_out = seg_model.segment(embeddings)

    inputs_for_prompt = []
    number_of_tokens = []

    helper_index = 0

    for utterance_ind in range(len(conv)):
        # finish always at Helper's utterance
        if speaker[utterance_ind] == 'Seeker':
            continue
        # go back to the segment beginning and capture previous segment
        input_for_prompt = []
        count_segments = 0
        pointer = utterance_ind
        while count_segments < nr_segments_to_include and pointer >= 0:
            count_segments += segmentation_out[pointer]
            input_for_prompt.append(f'{speaker[pointer]}: {conv[pointer]}')
            pointer -= 1
        input_for_prompt.reverse()
        inputs_for_prompt.append({"input": "\n".join(input_for_prompt),
                                  "conv_index": conv_index, "helper_index": helper_index})
        number_of_tokens.append(num_tokens_from_string(' '.join(input_for_prompt), 'cl100k_base'))
        helper_index += 1
    return inputs_for_prompt, number_of_tokens


def create_prompt_inputs_for_gpt(conv_index, conv, speaker):
    """Returns a list of gpt prompt inputs and a list of their lengths [nr tokens] for a conversation.
    It segments the conversation, first and last segments are merged to the following/previous segment.
    Then input context is 5 helpers responses"""

    inputs_for_prompt = []
    number_of_tokens = []
    helper_map_to_index = []
    helper_indices = []
    helper_index = -1
    for i, s in enumerate(speaker):
        if s == 'Helper':
            helper_index += 1
            helper_map_to_index.append(helper_index)
            helper_indices.append(i)
        else:
            helper_map_to_index.append(-1)

    new_segment_starts = []
    # start every third helper
    for helper_index in helper_map_to_index:
        if helper_index % 3 == 0:
            # add extra two for context, the annotations of those will be dropped, see annotate_with_gpt
            new_segment_starts.append(helper_indices[max(0, helper_index-2)])

    for seg_start_ind in new_segment_starts:
        # attach all utterances from the segment until there are 5 helper's responses
        helper_count = 0
        total_helpers = 5
        if seg_start_ind == new_segment_starts[0]:
            total_helpers = 3
        input_for_prompt = []
        helper_indices = []
        pointer = seg_start_ind

        # the segment starts with helper, attach previous utterance
        if speaker[pointer] == 'Helper' and pointer > 0:
            input_for_prompt.append(f'{speaker[pointer-1]}: {conv[pointer-1]}')

        input_for_prompt.append(f'{speaker[pointer]}: {conv[pointer]}')
        if helper_map_to_index[pointer] != -1:
            helper_indices.append(helper_map_to_index[pointer])
            helper_count += 1
        pointer += 1
        while pointer < len(conv) and helper_count < total_helpers:
            input_for_prompt.append(f'{speaker[pointer]}: {conv[pointer]}')
            if helper_map_to_index[pointer] != -1:
                helper_indices.append(helper_map_to_index[pointer])
                helper_count += 1
            pointer += 1
        inputs_for_prompt.append({"input": "\n".join(input_for_prompt),
                                  "conv_index": conv_index, "helper_indices": helper_indices})
        number_of_tokens.append(num_tokens_from_string(' '.join(input_for_prompt), 'cl100k_base'))
    return inputs_for_prompt, number_of_tokens



# ------------------------------------- PLOTTING ----------------------------------------------- #

def plot_histogram(total_stats, xlabel, ylabel, title, save_path, bins=None):

    if bins is None:
        bins = int(np.sqrt(len(total_stats)))

    for type in ['.pdf', '.png']:
        sns.set_context("paper")
        sns.set_theme(font='Times New Roman', font_scale=1.2, style='darkgrid', palette="pastel")
        plt.clf()
        hist = sns.histplot(total_stats, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # plt.show()
        hist.get_figure().savefig(f'{save_path}{type}')


def plot_tokens_in_esconv():
    dataset = load_esconv_dataset()
    for split in ['train', 'test']:
        tokens = []
        for ind in range(len(dataset[split])):
            datapoint = dataset[split][ind]
            conv, speaker = process_datapoint(datapoint)
            total_tokens = 0
            # merge speaker and conv
            for i in range(len(conv)):
                total_tokens += num_tokens_from_string(f'{speaker[i]}: {conv[i]}', 'cl100k_base')
            tokens.append(total_tokens)

        for suffix in ['.pdf', '.png']:
            plot_histogram(tokens, "Number of tokens", "Number of conversations",
                           f"Distribution of tokens in ESConv {split}",
                           f"plots/token_distribution_{split}{suffix}")


# ------------------------------------- PRETTY PRINTING ----------------------------------------------- #


def print_conv(ind):
    dataset = load_esconv_dataset()
    datapoint = dataset['train'][ind]
    conv, speaker = process_datapoint(datapoint)

    for i in range(len(conv)):
        print(f'{speaker[i]}: {conv[i]}')


def pretty_print_splitted_conv(ind):
    dataset = load_esconv_dataset()
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seg_model = C99()

    datapoint = dataset['train'][ind]
    conv, speaker = process_datapoint(datapoint)
    segmentation_out = seg_model.segment(conversation_to_embeddings(emb_model, conv))

    splitted_conv = []
    last_segment = []
    for i, switch_indicator in enumerate(segmentation_out):
        if switch_indicator == 1:
            splitted_conv.append(last_segment)
            last_segment = []
        last_segment.append(conv[i])
    splitted_conv.append(last_segment)
    splitted_conv = splitted_conv[1:]

    def prCyan(skk):
        print("\033[96m {}\033[00m".format(skk))

    ind = 0
    for segment in splitted_conv:
        for line in segment:
            if speaker[ind] == 'Helper':
                prCyan('Helper: ' + line)
            else:
                print('Seeker: ' + line)
            ind += 1
        print('\n-------------------\n')

# ------------------------------------- HUMAN ANNOTATION UTILS ----------------------------------------------- #
# the functions below were used during data processing stage using raw data
# merge_datasets_range_10("<expert annotations>", 0, 400, clean_gpt4=True)
# create_train_dataset(), create_test_dataset(), create_dataset_pre_dpo()
# boosted_vs_non_boosted('<sft generations path>')
# create_dpo_dataset('<sft generations path>_non_boosted.json', '<sft generations path>_boosted.json')
# make_ablations_data_dataset(), make_ablations_preference_dataset()
def add_alternatives():

    with open('<expert annotations file>') as json_file:
        data = json.load(json_file)
        new_data = []
        for datapoint in data:
            new_data.append(datapoint)
            if datapoint['output']['perfect'] is False:
                new_entry = {}
                new_entry['instruction'] = datapoint['instruction']
                new_entry['input'] = '\n'.join(datapoint['input'].split('\n')[:-1]) + '\n' + 'Helper: ' + datapoint['output']['alternative']
                new_entry['output'] = {'helper': 'Helper: ' + datapoint['output']['alternative'], 'perfect': True}
                new_entry['helper_index'] = datapoint['helper_index']
                new_entry['conv_index'] = datapoint['conv_index']
                good_areas = [ba for ba in datapoint['output']['badareas']]
                if 'Suggestions' in good_areas:
                    good_areas.remove('Suggestions')
                for original_good_area in datapoint['output']['goodareas']:
                    if original_good_area not in good_areas:
                        good_areas.append(original_good_area)
                new_entry['output']['goodareas'] = good_areas
                new_data.append(new_entry)

        for i, ann in enumerate(new_data):
            if 'badareas' in ann['output']:
                for badarea in ann['output']['badareas']:
                    if badarea in ann['output']['goodareas']:
                        ann['output']['goodareas'].remove(badarea)

        # shuffle new_data
        random.seed(42)
        random.shuffle(new_data)
        # save to json
        with open('<data with alternatives file>', 'w') as outfile:
            json.dump(new_data, outfile)


def create_train_dataset():
    merge_datasets_range_10("<expert annotation files with alternatives>", 0, 400, clean_gpt4=True)
    # read json
    with open('<merged expert annotation files>') as json_file:
        # save it as train.json
        data = json.load(json_file)
        add_alternatives()

    output = []
    with open(f'<data with alternatives>') as json_file:
        data = json.load(json_file)
        for ann in data:
            out = {}
            out['text'] = generate_prompt_with_response(ann)
            out['helper_index'] = ann['helper_index']
            out['conv_index'] = ann['conv_index']
            output.append(out)
        # save to json
        with open('<feedback qesconv train data>', 'w') as outfile:
            json.dump(output, outfile)

def make_ablations_data_dataset():
    with open('<sft generations>') as json_file:
        data = json.load(json_file)
        output = []
        for ann in data:
            out = {}
            out['text'] = ann['prompt'][3:] + json.dumps(ann['output'])
            out['helper_index'] = ann['helper_index']
            out['conv_index'] = ann['conv_index']
            output.append(out)
        # save to json
        with open('<ablation data>', 'w') as outfile:
            json.dump(output, outfile)

def make_ablations_preference_dataset():

    with open('<dpo train>') as json_file:
        data = json.load(json_file)
        output = []
        for ann in data:
            out = {}
            out['text'] = ann['prompt'] + ann['chosen']
            out['helper_index'] = ann['helper_index']
            out['conv_index'] = ann['conv_index']
            output.append(out)
        with open('<dpo pre data>', 'w') as outfile:
            json.dump(output, outfile)

def create_test_dataset():
    preprocess_dataset('test', gpt=False, save_suffix=f"_{0}-{len(load_esconv_dataset()['test'])}", indices_range=[i for i in range(len(load_esconv_dataset()['test']))])
    with open(f'prompts/prompts_input_part_test_0-{len(load_esconv_dataset()["test"])}.json') as json_file:
        data = json.load(json_file)
        output = []
        for pr in data:
            out = {}
            pr['instruction'] = "Give feedback to the Helper's last response."
            out['text'] = generate_prompt(pr)
            out['helper_index'] = pr['helper_index']
            out['conv_index'] = pr['conv_index']
            output.append(out)
        with open('<feedback quesconv test data>', 'w') as outfile:
            json.dump(output, outfile)


def create_dataset_pre_dpo():
    assert DATASET == 'qesconvdpo'
    for i in range(0, 150, 10):
        preprocess_dataset('train', gpt=False, save_suffix=f"DPO_{i}-{i+9}", indices_range=[i for i in range(i, i+10)])

    merge_datasets_range_10("<DPO data prompts>", 0, 150, clean_gpt4=False)

    # read json file
    with open('<merged DPO data prompts>') as json_file:
        data = json.load(json_file)
        output = []
        for pr in data:
            out = {}
            pr['instruction'] = "Give feedback to the Helper's last response."
            out['text'] = generate_prompt(pr)
            out['helper_index'] = pr['helper_index']
            out['conv_index'] = pr['conv_index']
            output.append(out)
        with open('<dpo pre data>', 'w') as outfile:
            json.dump(output, outfile)

def boosted_vs_non_boosted(generations_file, all=False):

    diff = []

    # if all, if no booster, attach the same annotation to two files (needed for interface comparison, not needed for dpo)
    with open(generations_file) as json_file:
        data = json.load(json_file)
        boosted = []
        non_boosted = []
        for ann in data:
            first_level_prob = ann['prob']
            # find max and min scores for second-level
            max_score_ind = -1
            max_score = -1
            min_score_ind = -1
            min_score = 2
            for ind, sample in enumerate(ann['output']):
                if sample['perfect'] is False:
                    score = sample['improved']['prob']
                    if score > max_score:
                        max_score = score
                        max_score_ind = ind
                    if score < min_score:
                        min_score = score
                        min_score_ind = ind

            added = False

            # if prob <= 0.5 this is negative feedback, optimize it's quality through self-scoring
            if first_level_prob <= 0.5:
                if max_score_ind != -1 and min_score_ind != -1:
                    diff.append(max_score - min_score)
                    boosted.append({'prompt': ann['prompt'], 'output': ann['output'][max_score_ind],
                                    'conv_index': ann['conv_index'], 'helper_index': ann['helper_index']})
                    non_boosted.append({'prompt': ann['prompt'], 'output': ann['output'][min_score_ind],
                                    'conv_index': ann['conv_index'], 'helper_index': ann['helper_index']})
                    added = True

            if all and not added:
                # add first generation to boosted and non-boosted
                boosted.append({'prompt': ann['prompt'], 'output': ann['output'][0],
                                'conv_index': ann['conv_index'], 'helper_index': ann['helper_index']})
                non_boosted.append({'prompt': ann['prompt'], 'output': ann['output'][0],
                                    'conv_index': ann['conv_index'], 'helper_index': ann['helper_index']})


            if len(boosted) > 0:
                if 'improved' in boosted[-1]['output']:
                    del boosted[-1]['output']['improved']
                if 'improved' in non_boosted[-1]['output']:
                    del non_boosted[-1]['output']['improved']

        print(f'Boosted: {len(boosted)}')
        print(f'Non-boosted: {len(non_boosted)}')

        all_str = ""
        if all:
            all_str = " ALL"
        # save to json
        with open(f"{generations_file.split('.')[0]}_boosted{all_str}.json", 'w') as outfile:
            json.dump(boosted, outfile)
        with open(f"{generations_file.split('.')[0]}_non_boosted{all_str}.json", 'w') as outfile:
            json.dump(non_boosted, outfile)


def create_dpo_dataset(non_boosted_file, boosted_file):
    # open two files
    with open(non_boosted_file) as json_file:
        non_boosted = json.load(json_file)
    with open(boosted_file) as json_file:
        boosted = json.load(json_file)

    output_train = []
    output_test = []
    for ind, ann in enumerate(boosted):
        new_entry = {}
        new_entry['prompt'] = ann['prompt']
        new_entry['chosen'] = json.dumps(boosted[ind]['output'])
        new_entry['rejected'] = json.dumps(non_boosted[ind]['output'])
        new_entry['conv_index'] = boosted[ind]['conv_index']
        new_entry['helper_index'] = boosted[ind]['helper_index']
        assert boosted[ind]['conv_index'] == non_boosted[ind]['conv_index']
        assert boosted[ind]['helper_index'] == non_boosted[ind]['helper_index']

        if new_entry['chosen'] != new_entry['rejected']:
            if new_entry['conv_index'] < 135:
                output_train.append(new_entry)
            if new_entry['conv_index'] >= 135:
                output_test.append(new_entry)

    with open('<dpo train file>', 'w') as outfile:
        json.dump(output_train, outfile)
    with open('<dpo test file>', 'w') as outfile:
        json.dump(output_test, outfile)


# ------------------------------------- PROCESSING UTILS ----------------------------------------------- #

def preprocess_dataset(split='train', gpt=False, save_suffix = "", indices_range=None):
    """Creates prompt inputs for a given split of the dataset and saves them to a json file.
    :param split: train or test
    :param gpt: if True, creates prompts for GPT models, otherwise for fine-tuning Llama
    """


    if not os.path.exists('plots'):
        os.makedirs('plots')


    print("------------------- Processing dataset to create prompts. ----------------------")

    # iterate over all datapoints in the training dataset
    dataset = load_esconv_dataset()
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    seg_model = C99()

    total_tokens = []
    all_prompts = []
    if indices_range is None:
        indices_range = range(len(dataset[split]))
    for ind in tqdm(indices_range, desc=f"[{split}] Generating input prompts"):
        datapoint = dataset[split][ind]
        conv, speaker = process_datapoint(datapoint)

        # create prompt inputs
        if gpt:
            inputs_for_prompt, number_of_tokens = create_prompt_inputs_for_gpt(conv_index=ind, conv=conv,
                                                                               speaker=speaker)
        else:
            inputs_for_prompt, number_of_tokens = create_prompt_inputs(emb_model=emb_model, seg_model=seg_model,
                                                                       conv_index=ind, conv=conv, speaker=speaker)
        total_tokens.extend(number_of_tokens)

        for inp in inputs_for_prompt:
            all_prompts.append(inp)

    gpt_str = "_GPT" if gpt else ""

    print(f"Saving prompts to prompts/prompts_input_part_{split}{gpt_str}{save_suffix}.json")
    with open(f'prompts/prompts_input_part_{split}{gpt_str}{save_suffix}.json', 'w') as f:
        json.dump(all_prompts, f)

    print("Total number of tokens in prompts:", sum(total_tokens),
          "\nTotal number of prompts:", len(total_tokens))

    print(f"Saving histogram to plots/plots/token_distribution_prompt_input_{split}{gpt_str}{save_suffix}.pdf")


    plot_histogram(total_tokens, "Number of tokens", "Number of prompts",
                   "Distribution of number of tokens in prompts (input part)",
                   f"plots/token_distribution_prompt_input_{split}{gpt_str}{save_suffix}")

    print("------------------- Processing finished. ----------------------")


def merge_json_datasets(datasets_list, save_path, clean_gpt4=False):
    """Merges multiple json datasets into one and saves it to a file."""
    merged_dataset = []
    for dataset in datasets_list:
        with open(dataset) as json_file:
            data = json.load(json_file)
            if clean_gpt4:
                data = clean_gpt4_annotations(data)
            merged_dataset.extend(data)

    # print length of merged dataset:
    print(f"Length of merged dataset: {len(merged_dataset)}")

    with open(save_path, 'w') as outfile:
        json.dump(merged_dataset, outfile)


def merge_datasets_range_10(dataset_path_root, range_start=0, range_end=50, clean_gpt4=False):
    """Merges multiple json datasets into one and saves it to a file."""

    dataset_list = []
    for i in range(range_start, range_end, 10):
        dataset_list.append(dataset_path_root + f"_{i}-{i+9}.json")

    print(dataset_path_root + f"_merged_{range_start}-{range_end-1}_cleaned_{clean_gpt4}.json")
    merge_json_datasets(dataset_list, dataset_path_root + f"_merged_{range_start}-{range_end-1}_cleaned_{clean_gpt4}.json", clean_gpt4)


def clean_gpt4_annotations(data):
    """Removes unncessary keys from the dataset."""

    for i in range(len(data)):
        # find all keys in i
        keys = list(data[i]["output"].keys())
        if data[i]["output"]["perfect"] == True:
            required_keys = ["helper", "perfect", "goodareas"]
        else:
            required_keys = ["helper", "perfect", "goodareas", "feedback", "badareas", "alternative"]

        for k in keys:
            if k not in required_keys:
                del data[i]["output"][k]

        if 'gpt_prompt_input' in data[i]:
            del data[i]['gpt_prompt_input']

    # return cleaned dataset
    return data



def count_utterances_and_words(split='train', speaker_selection=None):

    # make plots directory
    if not os.path.exists('plots'):
        os.makedirs('plots')

    dataset = load_esconv_dataset()
    number_of_utterances = []
    number_of_words = []
    words_per_utterance = []
    length_to_ind = {}
    for ind, datapoint in enumerate(dataset[split]):

        conv, speaker = process_datapoint(datapoint)

        if speaker_selection is not None:
            conv = [conv[i] for i in range(len(conv)) if speaker[i] == speaker_selection]

        number_of_utterances.append(len(conv))
        number_of_words.append(sum([len(line.split()) for line in conv]))
        words_per_utterance.extend([len(line.split()) for line in conv])

        if len(conv) not in length_to_ind:
            length_to_ind[len(conv)] = []
        length_to_ind[len(conv)].append(ind)

    # sort dictionary lenght_to_ind by key
    length_to_ind = {k: v for k, v in sorted(length_to_ind.items(), key=lambda item: item[0])}

    # count number of convs of each length
    length_to_count = {}
    for k, v in length_to_ind.items():
        length_to_count[k] = len(v)

    # save as json
    speaker_str = "" if speaker_selection is None else f"_{speaker_selection}"
    with open(f'plots/length_to_ind_{split}{speaker_str}.json', 'w') as outfile:
        json.dump(length_to_ind, outfile)

    with open(f'plots/length_to_count_{split}{speaker_str}.json', 'w') as outfile:
        json.dump(length_to_count, outfile)


    plot_histogram(number_of_utterances, "Number of utterances", "Number of conversations",
                   f"Distribution of number of utterances in {split} dataset", save_path=f"plots/utterances_{split}{speaker_str}")
    plot_histogram(number_of_words, "Number of words per conversation", "Number of conversations",
                   f"Distribution of total number of words in {split} dataset", save_path=f"plots/words_{split}{speaker_str}")
    plot_histogram(words_per_utterance, "Number of words per utterance", "Number of utterances",
                     f"Distribution of number of words per utterance in {split} dataset", save_path=f"plots/words_per_utterance_{split}{speaker_str}")

    if DATASET == 'qesconv':
        plot_histogram(number_of_utterances, "Number of utterances", "Number of conversations", f"", save_path=f"plots/qesconv_utterances_{speaker_str}")
        plot_histogram(words_per_utterance, "Number of words per utterance", "Number of utterances", f"", save_path=f"plots/qesconv_words_per_utterance_{speaker_str}")

    print(f'Total number of utterances {np.sum(number_of_utterances)}')
    print(f'Average number of utterances per conversation: {np.mean(number_of_utterances)}')
    print(f'Average number of words per conversation: {np.mean(number_of_words)}')
    print(f'Average number of words per utterance: {np.mean(words_per_utterance)}')

# ------------------------------------- prompting utils ----------------------------------

def generate_prompt(example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    )

def generate_prompt_with_response(example):
    # remove helper key
    del example['output']['helper']
    return generate_prompt(example) + json.dumps(example["output"])



if __name__ == '__main__':

    # utils.py contain functions we used during data processing/cleaning

    # to create prompts for gpt annotation run
    # for i in range(0, 400, 10):
    #     preprocess_dataset('train', gpt=True, save_suffix=f"_{i}-{i+9}", indices_range=[i for i in range(i, i+10)])
    #     preprocess_dataset('train', gpt=False, save_suffix=f"_{i}-{i + 9}",indices_range=[i for i in range(i, i + 10)])

    # print example train conversation
    # pretty_print_splitted_conv(78)

    # see qesconv stats
    # count_utterances_and_words(split='train')
    # count_utterances_and_words(split='train', speaker_selection='Helper')
    # count_utterances_and_words(split='train', speaker_selection='Seeker')

    pass

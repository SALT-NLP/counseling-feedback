import seaborn as sns
import json
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats
import pandas as pd
import pingouin as ping

file_mapping = {"Self-perfecting": "output/feedback_qesconv_dpo_model_generations_0_sft_model_probs.json",
                "SFT": "output/feedback_qesconv_sft_model_generations_0_sft_model_probs.json",
                "+ generations": "output/feedback_qesconv_ablation_data_model_generations_0_sft_model_probs.json",
                "+ best generations": "output/feedback_qesconv_ablation_preference_model_generations_0_sft_model_probs.json", }

def get_scores(dist, worst_percent=0.1):

    file_name = file_mapping[dist]
    with open(file_name) as f:
        data = json.load(f)

    prob_list = []
    for prompt in data:
        for output in prompt['output']:
            if 'improved' in output:
                prob_list.append(output['improved']['prob'])

    # return worst generations
    prob_list.sort()
    prob_list = prob_list[:int(len(prob_list)*worst_percent)]

    return prob_list

def plot_hist_two_samples(dist_1, dist_2, worst_percent=0.1, num_bins=30, ylim=0.33):
    scores_1 = get_scores(dist_1, worst_percent)
    scores_2 = get_scores(dist_2, worst_percent)

    col_mapping = {"Self-perfecting": "orange",
                    "SFT": "grey",
                    "+ generations": "violet",
                    "+ best generations": "turquoise", }
    col1 = col_mapping[dist_1]
    col2 = col_mapping[dist_2]

    # if Self-perfecting rename to Self-improved
    if dist_1 == "Self-perfecting":
        dist_1 = "Self-improved"
    if dist_2 == "Self-perfecting":
        dist_2 = "Self-improved"

    # if + generations rename to +new data
    if dist_1 == "+ generations":
        dist_1 = "+ new data"
    if dist_2 == "+ generations":
        dist_2 = "+ new data"

    # if + best generations rename to +best scores
    if dist_1 == "+ best generations":
        dist_1 = "+ best scores"
    if dist_2 == "+ best generations":
        dist_2 = "+ best scores"

    bins = np.linspace(0, 1, num_bins)  # num_bins is the number of bins you want

    sns.histplot(scores_1, color=col1, label=dist_1, kde=False, stat='probability', bins=bins, alpha=1.0)
    plt.ylim(0, ylim)
    sns.histplot(scores_2, color=col2, label=dist_2, kde=False, stat='probability', bins=bins, alpha=0.7)
    plt.ylim(0,ylim)
    plt.ylabel("Frequency")
    plt.title(f"Scores [worst {int(worst_percent*100)}%]")
    if worst_percent <= 0.01:
        loc='upper right'
    else:
        loc='upper left'
    plt.legend(loc=loc)

def statistical_tests(dist_1, dist_2, worst_percent=0.1):
    print(f'\n\n{dist_1} vs {dist_2}')
    # print means
    print(f'{dist_1} mean: {np.mean(get_scores(dist_1, worst_percent))}')
    print(f'{dist_2} mean: {np.mean(get_scores(dist_2, worst_percent))}')

    scores_1 = get_scores(dist_1, worst_percent)
    scores_2 = get_scores(dist_2, worst_percent)
    print(stats.ranksums(scores_1, scores_2))
    print(stats.ttest_ind(scores_1, scores_2, equal_var=False))

if __name__ == '__main__':

    sns.set_theme("paper", style="white", font_scale=3.0, palette='pastel')
    plt.figure(figsize=(24, 14))
    sns.set_style({'font.family': 'Times New Roman'})
    plt.subplot(2, 3, 1)
    plot_hist_two_samples("SFT", "+ generations", 0.01, 10, 0.5)
    plt.grid(linestyle='dotted', axis='y')
    plt.subplot(2, 3, 2)
    plot_hist_two_samples("SFT", "+ best generations", 0.01, 10, 0.5)
    plt.grid(linestyle='dotted', axis='y')
    plt.subplot(2, 3, 3)
    plot_hist_two_samples("SFT", "Self-perfecting", 0.01, 10, 0.5)
    plt.grid(linestyle='dotted', axis='y')
    plt.subplot(2, 3, 4)
    plot_hist_two_samples("SFT", "+ generations", 0.05)
    plt.grid(linestyle='dotted', axis='y')
    plt.subplot(2, 3, 5)
    plot_hist_two_samples("SFT", "+ best generations", 0.05)
    plt.grid(linestyle='dotted', axis='y')
    plt.subplot(2, 3, 6)
    plot_hist_two_samples("SFT", "Self-perfecting", 0.05)
    plt.grid(linestyle='dotted', axis='y')

    ax = plt.gca()
    ax.set_axisbelow(True)
    sns.move_legend(ax, "upper left")
    plt.savefig("scores_overall.pdf")
    plt.show()

    statistical_tests("SFT", "+ generations", worst_percent=0.01)
    statistical_tests("SFT", "+ best generations", worst_percent=0.01)
    statistical_tests("SFT", "Self-perfecting", worst_percent=0.01)
    statistical_tests("SFT", "+ generations", worst_percent=0.05)
    statistical_tests("SFT", "+ best generations", worst_percent=0.05)
    statistical_tests("SFT", "Self-perfecting", worst_percent=0.05)
    statistical_tests("SFT", "+ generations", worst_percent=1)
    statistical_tests("SFT", "+ best generations", worst_percent=1)
    statistical_tests("SFT", "Self-perfecting", worst_percent=1)


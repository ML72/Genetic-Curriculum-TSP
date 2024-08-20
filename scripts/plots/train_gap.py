import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Define constants, modify as necessary to obtain different training gap plots
# Modify the PLOTS constant to generate different plots
# Modify the MODELS constant to change the plotted models
BASE_DIR = 'results/tsp'
RESULT_FILENAME = 'test.png' # 'train_gap_main.png', 'train_gap_appendix.png'
PLOTS = [
    {'dataset': 'tsp_gmm50_test_seed1234', 'field': 'Gap_Avg', 'name': 'Avg Gap'},
    {'dataset': 'tsp_diag50_test_seed1234', 'field': 'Gap_Avg', 'name': 'Avg Gap'},
    {'dataset': 'tsp_tsplib50_test_seed1234', 'field': 'Gap_Avg', 'name': 'Avg Gap'},
    {'dataset': 'tsp_tsplib50_test_seed1234', 'field': 'Gap_Worst_1.0', 'name': 'Worst 1% Gap'},
    {'dataset': 'tsp_tsplib50_test_seed1234', 'field': 'Gap_Worst_0.5', 'name': 'Worst 0.5% Gap'},
    {'dataset': 'tsp_tsplib50_test_seed1234', 'field': 'Gap_Worst_0.1', 'name': 'Worst 0.1% Gap'},
    #{'dataset': 'tsp_gmm50_test_seed1234', 'field': 'Gap_Worst_1.0', 'name': 'Worst 1% Gap'},
    #{'dataset': 'tsp_gmm50_test_seed1234', 'field': 'Gap_Worst_0.5', 'name': 'Worst 0.5% Gap'},
    #{'dataset': 'tsp_gmm50_test_seed1234', 'field': 'Gap_Worst_0.1', 'name': 'Worst 0.1% Gap'},
    #{'dataset': 'tsp_diag50_test_seed1234', 'field': 'Gap_Worst_1.0', 'name': 'Worst 1% Gap'},
    #{'dataset': 'tsp_diag50_test_seed1234', 'field': 'Gap_Worst_0.5', 'name': 'Worst 0.5% Gap'},
    #{'dataset': 'tsp_diag50_test_seed1234', 'field': 'Gap_Worst_0.1', 'name': 'Worst 0.1% Gap'},
]
MODELS = {
    'baseline_unif': 'Uniform (Baseline)',
    'hac_fullbatch': 'HAC (Baseline)',
    'curriculum_ewc_genome': 'Genetic Curriculum (Ours)',
    #'ablation_noewc': 'Ablation (No EWC)',
    #'ablation_nogenome': 'Ablation (No Genome)',
}
DATASETS = {
    'tsp_unif50_test_seed1234': "Uniform",
    'tsp_gmm50_test_seed1234': "Gaussian Mixture",
    'tsp_diag50_test_seed1234': "Diagonal",
    'tsp_tsplib50_test_seed1234': "TSPLib50"
}
COLORS = {
    'baseline_unif': 'purple',
    'hac_fullbatch': 'red',
    'curriculum_ewc_genome': 'blue',
    'ablation_noewc': 'green',
    'ablation_nogenome': 'orange'
}
WIDTH = 3
TRIALS = 5

# Utility function for reading a json file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Utility function for plotting a subplot of a certain graph
def subplot_embedding(subplot, dataset, field, title):
    # Collect data
    data = {}
    for model in MODELS.keys():
        for trial in range(1, TRIALS+1):
            file_path = f"{BASE_DIR}/{dataset}/{model}_{trial}-epoch_data.json"
            data[f"{model}_{trial}"] = read_json_file(file_path)

    epochs = data[list(data.keys())[0]].keys()
    epochs = [int(epoch) for epoch in epochs]
    epochs.sort()

    # Plot data
    y_max = 0
    for model in MODELS.keys():
        values = np.zeros((len(epochs), TRIALS))
        for i in range(len(epochs)):
            for j in range(TRIALS):
                values[i, j] = float(data[f"{model}_{j+1}"][str(epochs[i])][field])
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        subplot.plot(epochs, means, label=MODELS[model], color=COLORS[model])
        if not np.allclose(stds, 0):
            subplot.fill_between(epochs, means-stds, means+stds, color=COLORS[model], alpha=0.2)
        y_max = max(y_max, np.max(means[1:]))

    subplot.set_title(title)
    subplot.margins(x=0)
    subplot.xlim = (0, max(epochs))
    subplot.set_ylim(top=y_max * 1.1)
    subplot.set_xlabel('Epoch')
    subplot.set_ylabel('Gap (%)')

# Draw plot visualizations
height = math.ceil(len(PLOTS) / WIDTH)
fig, axs = plt.subplots(height, WIDTH, figsize=(3.5 * WIDTH, 3.5 * height))
plt.subplots_adjust(hspace=0.1)

for i, plot in enumerate(PLOTS):
    print(f"Plotting from {DATASETS[plot['dataset']]}")
    target_axs = axs[i] if height == 1 else axs[i // WIDTH, i % WIDTH]
    subplot_embedding(
        target_axs,
        plot['dataset'],
        plot['field'],
        f"{plot['name']} on {DATASETS[plot['dataset']]}"
    )

# Add custom legend for plot colors
custom_handles = [
    mpatches.Patch(color=COLORS[model], label=name) for model, name in MODELS.items()
]

fig.legend(handles=custom_handles, loc='upper center', ncol=4)
plt.tight_layout(rect=[0.025, 0.015, 0.95, 0.94])

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()

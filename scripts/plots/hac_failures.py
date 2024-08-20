import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Define constants
EVAL_PATH = 'results/tsp/tsp_tsplib50_test_seed1234/tsp_tsplib50_test_seed1234-tsp_50_hac_fullbatch_1-greedy-t1-0-10000.pkl'
ORACLE_TOURS_PATH = 'results/tsp/tsp_tsplib50_test_seed1234/concorde_tours.pkl'
DATASET_PATH = 'data/tsp/tsp_tsplib50_test_seed1234.pkl'
RESULT_FILENAME = 'prelim_hac_failures.png'
NUM_WORSTCASE = 4
STYLES = {
    'Model Path': {'color': 'blue', 'style': 'solid'},
    'Oracle Path': {'color': 'red', 'style': 'dashed'}
}

# Utility function for reading a pickle file
def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data

# Load data
tours, gaps = read_pickle_file(EVAL_PATH)
oracle_tours = np.array(read_pickle_file(ORACLE_TOURS_PATH))
dataset = np.array(read_pickle_file(DATASET_PATH))

# Utility function for plotting subplot
def subplot_embedding(subplot, idx):
    # Plot points
    graph = dataset[idx]
    num_points = len(graph)
    subplot.scatter(graph[:,0], graph[:,1], color='black')

    # Plot connections between points
    model_tour = np.array(tours[idx][1]).astype(int)
    oracle_tour = oracle_tours[idx].astype(int)
    for i in range(num_points):
        i_next = (i + 1) % num_points
        subplot.plot(
            [graph[oracle_tour[i]][0], graph[oracle_tour[i_next]][0]],
            [graph[oracle_tour[i]][1], graph[oracle_tour[i_next]][1]],
            color=STYLES['Oracle Path']['color'],
            linestyle=STYLES['Oracle Path']['style'],
            linewidth=0.75
        )
        subplot.plot(
            [graph[model_tour[i]][0], graph[model_tour[i_next]][0]],
            [graph[model_tour[i]][1], graph[model_tour[i_next]][1]],
            color=STYLES['Model Path']['color'],
            linestyle=STYLES['Model Path']['style'],
            linewidth=0.75
        )

    # Set style
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title(f'Gap: {np.round(gaps[idx], 2)}%')

# Plot worst-case instances
worstcase_indices = np.argsort(gaps)[-NUM_WORSTCASE:]
fig, axs = plt.subplots(1, NUM_WORSTCASE, figsize=(NUM_WORSTCASE * 4, 4))
plt.subplots_adjust(hspace=0.1)
plt.tight_layout(rect=[0.01, 0, 0.97, 0.99])

for i in range(NUM_WORSTCASE):
    subplot_embedding(axs[i], worstcase_indices[i])

# Add custom legend for plot colors
custom_handles = [
    mpatches.Patch(
        color=style['color'], label=name, linestyle=style['style']
    ) for name, style in STYLES.items()
]

fig.legend(handles=custom_handles, loc='upper center', ncol=4)
plt.tight_layout(rect=[0.025, 0.015, 0.95, 0.94])

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()

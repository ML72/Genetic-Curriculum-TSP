import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



# Define constants
RESULT_FILENAME = 'distribution_visualization.png'
DISTRIBUTIONS = {
    'TSPLib50': 'data/tsplib',
    'Gaussian Mixture': 'data/tsp/tsp_gmm50_test_seed1234',
    'Diagonal': 'data/tsp/tsp_diag50_test_seed1234'
}
NUM_SAMPLES = 2
NUM_NODES = 50
IDX_LIST = {
    'TSPLib50': [4, 6],
    'Gaussian Mixture': [1500, 3500],
    'Diagonal': [5000, 9000]
}

# Utility function for reading a pickle file
def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        pickle_data = pickle.load(f)
    return np.array(pickle_data)

# Load in TSPLib data
tsplib = []
for filename in os.listdir(DISTRIBUTIONS['TSPLib50']):
    if filename.endswith(".npy") and not filename.endswith("sol.npy"):
        instance = np.load(f"{DISTRIBUTIONS['TSPLib50']}/{filename}")
        if len(instance) >= 150:
            tsplib.append(instance)

# Load in all data
data = {}
for distribution in DISTRIBUTIONS.keys():
    if distribution == 'TSPLib50':
        data[distribution] = tsplib
    else:
        file_path = f"{DISTRIBUTIONS[distribution]}.pkl"
        data[distribution] = read_pickle_file(file_path)

# Utility function for plotting subplot
def subplot_embedding(subplot, graph, title):
    if graph.shape[0] <= NUM_NODES:
        subplot.scatter(graph[:,0], graph[:,1], color='black')
    else:
        np.random.seed(4321)
        idx = np.random.choice(graph.shape[0], NUM_NODES, replace=False)
        subplot.scatter(graph[:,0], graph[:,1], color='gray', alpha=0.25)
        subplot.scatter(graph[idx,0], graph[idx,1], color='black')
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title(title)

# Plot it!
fig, axs = plt.subplots(NUM_SAMPLES, 3, figsize=(3 * 3, NUM_SAMPLES * 3))
plt.subplots_adjust(hspace=0.1)
plt.tight_layout(rect=[0.01, 0, 0.97, 0.98])

for j, distribution in enumerate(DISTRIBUTIONS.keys()):
    for i in range(NUM_SAMPLES):
        idx = IDX_LIST[distribution][i]
        subplot_embedding(axs[i, j], data[distribution][idx], f"{distribution} Example {i+1}")

if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()

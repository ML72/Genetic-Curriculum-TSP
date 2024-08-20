import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt



# Define constants
EVAL_PATH = 'results/tsp/tsplib/tsplib-tsp_50_hac_fullbatch_1-greedy-t1-0-59.pkl'
RESULT_FILENAME = 'prelim_hac_lengthgen.png'
DATASET_PATH = 'data/tsplib'

# Load data
with open(EVAL_PATH, 'rb') as f:
    _, gaps = pkl.load(f)

lengths = []
for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".npy") and not filename.endswith("sol.npy"):
        lengths.append(np.load(f"{DATASET_PATH}/{filename}"))
lengths = np.array([len(l) for l in lengths])

# Print correlation coefficient
print("Correlation coefficient:")
print(np.corrcoef(lengths, gaps))

# Plot lengths to gaps
plt.figure().set_figheight(3)
plt.figure().set_figwidth(5)
plt.scatter(lengths, gaps, color='blue')
plt.title('HAC Gap (%) to TSPLib Instance Size')
plt.xlabel('TSPLib Instance Size')
plt.ylabel('Gap (%)')
plt.xscale('log')

# Save plot
if not os.path.exists("results/plots"):
    os.makedirs("results/plots")
plt.savefig(f"results/plots/{RESULT_FILENAME}", format='png')
plt.close()
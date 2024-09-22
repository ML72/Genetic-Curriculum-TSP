import numpy as np
import pickle as pkl
import os



# Define constants, modify fields as necessary
DATASET_PATH = 'data/tsp/tsp_tsplib50_test_seed1234.pkl'
RESULT_FILENAME = 'llm_prompts.txt'

# Load in data
with open(DATASET_PATH, 'rb') as f:
    dataset = pkl.load(f)

# Utility method for writing a TSP level to the result file
template = "There are {} cities, respectively at the following locations on a 2D plane:\n{}" \
        + "\nWhat is the optimal tour permutation of the cities to minimize the total distance traveled?" \
        + "\nSolve the problem to the best of your ability. Reply with only a permutation of the indices 1 to {}, and nothing else.\n"
def write_level(f, level):
    tsp_size = len(level)
    tsp_locations = ", ".join([f"({round(level[i][0], 5)}, {round(level[i][1], 5)})" for i in range(tsp_size)])
    f.write(template.format(tsp_size, tsp_locations, tsp_size))

# Write prompts to result file
if not os.path.exists("results/prompts"):
    os.makedirs("results/prompts")

with open(f"results/prompts/{RESULT_FILENAME}", 'w') as f:
    for i in range(10):
        if i > 0:
            f.write("\n\n\n")
        write_level(f, dataset[i])

print(f"Prompts written to results/prompts/{RESULT_FILENAME}")

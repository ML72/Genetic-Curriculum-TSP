import scipy.stats as stats
import numpy as np
import json



# Define constants, model fields as necessary to conduct different tests
DATASET_DIR = 'results/tsp/tsp_tsplib50_test_seed1234'
MODEL1 = 'hac_fullbatch'
MODEL2 = 'curriculum_ewc_genome'
FIELD = 'Gap_Worst_0.1'
TRIALS = 5

# Utility function for reading a json file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Read in data
results1 = np.zeros(TRIALS)
results2 = np.zeros(TRIALS)
for i in range(TRIALS):
    data1 = read_json_file(f"{DATASET_DIR}/{MODEL1}_{i+1}-epoch_data.json")
    data2 = read_json_file(f"{DATASET_DIR}/{MODEL2}_{i+1}-epoch_data.json")
    epochs_max1 = max([int(epoch) for epoch in data1.keys()])
    epochs_max2 = max([int(epoch) for epoch in data2.keys()])
    results1[i] = float(data1[str(epochs_max1)][FIELD])
    results2[i] = float(data2[str(epochs_max2)][FIELD])

# Conduct t-test
t_stat, p_val = stats.ttest_ind(results1, results2, equal_var=False)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")

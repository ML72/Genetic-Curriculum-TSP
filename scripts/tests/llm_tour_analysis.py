import numpy as np
import pickle as pkl



# Define constants, simply modify TOUR variable to the LLM output
TOUR_LIST = [
    [1, 46, 18, 27, 31, 15, 36, 37, 30, 29, 33, 19, 9, 3, 7, 38, 11, 13, 25, 34, 10, 6, 48, 24, 23, 28, 17, 21, 16, 42, 26, 39, 44, 20, 49, 47, 41, 14, 22, 32, 35, 40, 2, 8, 12, 45, 5, 43, 4, 50],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [1, 43, 26, 48, 29, 10, 11, 27, 9, 44, 21, 32, 40, 50, 5, 47, 13, 34, 36, 17, 7, 37, 41, 31, 3, 38, 2, 15, 46, 6, 19, 28, 4, 20, 24, 25, 16, 39, 42, 30, 35, 14, 18, 8, 23, 45, 49, 22, 33, 12],
    [1, 9, 19, 40, 11, 10, 36, 27, 38, 49, 41, 2, 16, 35, 20, 46, 18, 14, 33, 4, 7, 23, 6, 43, 44, 29, 8, 39, 24, 5, 21, 47, 30, 42, 22, 50, 28, 32, 45, 34, 31, 17, 48, 3, 15, 25, 12, 37, 26, 13],
    [1, 7, 10, 20, 14, 19, 23, 35, 3, 13, 43, 50, 2, 32, 11, 5, 16, 49, 48, 29, 28, 26, 22, 41, 47, 44, 18, 21, 31, 12, 46, 6, 42, 40, 38, 33, 30, 25, 39, 17, 24, 15, 36, 4, 8, 45, 9, 27, 37, 34]
]
IDX_LIST = [0, 1, 2, 3, 4]
DATASET_PATH = 'data/tsp/tsp_tsplib50_test_seed1234.pkl'
ORACLE_PATH = 'results/tsp/tsp_tsplib50_test_seed1234/concorde_costs.pkl'

# Load in data
with open(DATASET_PATH, 'rb') as f:
    dataset = pkl.load(f)
    dataset = np.array(dataset)
with open(ORACLE_PATH, 'rb') as f:
    oracle = pkl.load(f)
    oracle = np.array(oracle)

# Utility functions
def compute_distance(positions, tour):
    assert len(tour) == len(set(tour)), "Tour contains duplicates"
    distance = 0
    for i in range(len(tour) - 1):
        distance += np.linalg.norm(positions[tour[i]] - positions[tour[i + 1]])
    distance += np.linalg.norm(positions[tour[-1]] - positions[tour[0]])
    return distance

def optimality_gap(oracle_cost, tour_cost):
    return (tour_cost - oracle_cost) / oracle_cost

# Analyze tours
num_tours = len(TOUR_LIST)
found_costs = []
oracle_costs = []
gaps = []
for i in range(num_tours):
    idx = IDX_LIST[i]
    tour = np.array(TOUR_LIST[i]) - 1
    distance = compute_distance(dataset[idx], tour)
    gap = optimality_gap(oracle[idx], distance) * 100

    gaps.append(gap)
    found_costs.append(distance)
    oracle_costs.append(oracle[idx])

    print(f"Tour {i + 1}")
    print(f"Distance of tour {i + 1}: {distance}")
    print(f"Optimal distance: {oracle[idx]}")
    print(f"Optimality gap: {gap}%")
    print()

print(f"Average optimality gap: {np.mean(gaps)}%")
print(f"Average distance: {np.mean(found_costs)}")
print(f"Average optimal distance: {np.mean(oracle_costs)}")
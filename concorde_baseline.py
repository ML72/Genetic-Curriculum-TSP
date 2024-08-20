
import numpy as np
import torch
import pickle
import os
import argparse
from concorde.tsp import TSPSolver

from utils.local_search import tsp_length_batch


PRECISION_SCALAR = 100_000 # Concorde rounds edge weights
EDGE_WEIGHT_TYPE = "EUC_2D"

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Filename of the dataset to evaluate")
    opts = parser.parse_args()

    assert opts.data_path is not None, "Need to specify data path"

    # Read in dataset
    name = os.path.splitext(os.path.basename(opts.data_path))[0]
    with open(opts.data_path, "rb") as f:
        dataset = pickle.load(f)
    positions = np.array(dataset)

    # Solve each instance
    M, N, _ = positions.shape
    tours = np.zeros((M, N))

    for i in range(M):
        solver = TSPSolver.from_data(
            positions[i,:,0] * PRECISION_SCALAR,
            positions[i,:,1] * PRECISION_SCALAR,
            norm=EDGE_WEIGHT_TYPE
        )
        solution = solver.solve()
        tours[i] = solution.tour
        
        print(f"Finished instance {i+1}/{M} with cost {solution.optimal_value/PRECISION_SCALAR}")
        del solver
        del solution

    costs = tsp_length_batch(
        torch.from_numpy(positions),
        torch.from_numpy(tours).long()
    )

    # Write to output
    result_dir = f"results/tsp/{name}"
    os.makedirs(result_dir, exist_ok=True)

    costs = costs.numpy().tolist()
    with open(f"{result_dir}/concorde_costs.pkl", "wb") as f:
        pickle.dump(costs, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved concorde costs to {result_dir}/concorde_costs.pkl")

    tours = tours.tolist()
    with open(f"{result_dir}/concorde_tours.pkl", "wb") as f:
        pickle.dump(tours, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved concorde tours to {result_dir}/concorde_tours.pkl")

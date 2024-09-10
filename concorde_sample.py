
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
    parser.add_argument("--folder", type=str, default=None, help="Filename of the dataset to evaluate")
    parser.add_argument("--n_samples", type=str, default=None, help="Number of datapoints to run concorde on")
    opts = parser.parse_args()

    assert opts.folder is not None, "Need to specify folder"

    # Read in dataset
    basefolder = os.path.basename(opts.folder)
    # Initialize an empty list to store the .npy files
    npy_files = []

    # Loop through all the files in the directory
    for filename in os.listdir(basefolder):
        # Check if the file ends with .npy
        if filename.endswith('.npy'):
            # Add the .npy file to the list
            npy_files.append(filename)

            # Create a folder based on the npy file name (without extension)
            folder_name = filename.replace('.npy', '')  # Remove .npy extension
            output_folder = os.path.join(opts.folder, folder_name)

            # Create the directory for this npy file
            os.makedirs(output_folder, exist_ok=True)
            print(f"Folder created: {output_folder}")

    for npy_file in npy_files[::3]:

        npy_data = np.load(npy_file)
        folder_name = npy_file.replace('.npy', '')  # Remove .npy extension
        concorde_output_folder = os.path.join(opts.folder, folder_name+"_concorde")

        number_of_problems = npy_data.shape[0]
        chosen_indices = np.random.choice(number_of_problems, 10)

        chosen_problems = npy_data[chosen_indices]
        for problem, problem_idx in enumerate(chosen_problems, chosen_indices):

            positions = np.array(problem)

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
            os.makedirs(concorde_output_folder, exist_ok=True)

            costs = costs.numpy().tolist()
            with open(f"{concorde_output_folder}/concorde_costs_{problem_idx}.pkl", "wb") as f:
                pickle.dump(costs, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved concorde costs to {concorde_output_folder}/concorde_costs_{problem_idx}.pkl")

            tours = tours.tolist()
            with open(f"{concorde_output_folder}/concorde_tours_{problem_idx}.pkl", "wb") as f:
                pickle.dump(tours, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved concorde tours to {concorde_output_folder}/concorde_tours_{problem_idx}.pkl")

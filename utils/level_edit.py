import torch
import numpy as np


def global_perturb_tensor(tensor, percentage, max_perturb_degree=0.1):
    """
    Randomly perturbs a percentage of rows in a given tensor to varying degrees.
    """
    # Calculate the number of rows to perturb
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(tensor.size(0) * (percentage / 100))

    # Select random indices to perturb
    idx_perturb = np.random.choice(tensor.size(0), num_rows, replace=False)

    for idx in idx_perturb:
        # Generate a random perturbation magnitude between 0 and max_perturb_degree
        perturb_mag = torch.rand(1).item() * max_perturb_degree

        # Apply the perturbation with the generated magnitude
        perturbation = (torch.rand(tensor[idx].shape) - 0.5) * 2 * perturb_mag
        tensor[idx] = torch.clamp(tensor[idx] + perturbation, 0.0, 1.0)

    return tensor


def local_perturb_tensor(tensor, percentage, max_perturb_degree=0.1):
    """
    Randomly perturbs a sector of spatially close rows in a given tensor to varying degrees.
    """
    # Calculate the number of rows to perturb
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(tensor.size(0) * (percentage / 100))

    # Choose a random center point
    center_idx = np.random.choice(tensor.size(0), 1)[0]
    center_point = tensor[center_idx]

    # Calculate distances from the center point to all other points
    distances = torch.norm(tensor - center_point, dim=1)

    # Select the indices of the closest `num_rows` points
    idx_perturb = torch.argsort(distances)[:num_rows].tolist()

    for idx in idx_perturb:
        # Generate a random perturbation magnitude between 0 and max_perturb_degree
        perturb_mag = torch.rand(1).item() * max_perturb_degree

        # Apply the perturbation with the generated magnitude
        perturbation = (torch.rand(tensor[idx].shape) - 0.5) * 2 * perturb_mag
        tensor[idx] = torch.clamp(tensor[idx] + perturbation, 0.0, 1.0)

    return tensor


def random_edit_tensor(tensor, percentage):
    """
    Randomly edits a percentage of rows in a given tensor.
    """
    # Calculate the number of rows to edit
    assert 0 <= percentage <= 100, "Percentage must be between 0 and 100"
    num_rows = int(tensor.size(0) * (percentage / 100))

    # Select random indices to edit
    idx_edit = np.random.choice(tensor.size(0), num_rows, replace=False)

    for idx in idx_edit:
        # Edit the specified rows by replacing with random values
        tensor[idx] = torch.rand(tensor[idx].shape)

    return tensor

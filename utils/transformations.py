import torch


def fit_unit_batch(tensor):
    """
    Scales and translates a batch of coordinate tensors to maximally fit inside the unit square.

    Args:
    - tensor (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tensors
        in the batch and N is the number of points in each tensor.
    """
    tensor -= torch.min(tensor, dim=1, keepdim=True).values
    tensor /= torch.max(
        torch.max(
            tensor, dim=1, keepdim=True
        ).values, dim=2, keepdim=True
    ).values

    return tensor

def transform_tensor(tensor):
    """
    Rotates a coordinate tensor so that its first PC is along the (0,0)-(1,1) diagonal,
    and then scales and translates it to maximally fit inside the unit square.

    Args:
    - tensor (torch.Tensor): A 2D tensor of shape (N, 2) where N is the number of points.
    """
    # Compute the angle of the first principal component
    tensor_centered = tensor - torch.mean(tensor, dim=0)
    U, S, V = torch.linalg.svd(tensor_centered, full_matrices=False)
    angle = torch.atan2(V[0, 1], V[0, 0])

    # Calculate and apply the desired rotation matrix
    # Note the signs, this matrix is post-multiplied
    angle -= (torch.pi / 4)
    rotation = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ], device=tensor.device)

    rotated_coordinates = tensor @ rotation

    # Translate and scale points to fit inside unit square
    rotated_coordinates -= torch.min(rotated_coordinates, dim=0, keepdim=True).values
    rotated_coordinates /= torch.max(rotated_coordinates)

    return rotated_coordinates

def transform_tensor_batch(tensors):
    """
    Rotates a batch of coordinate tensors so that their first PCs are along the (0,0)-(1,1) diagonal,
    and then scales and translates them to each maximally fit inside the unit square.

    Args:
    - tensors (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tensors
        in the batch and N is the number of points in each tensor.
    """
    M, N, _ = tensors.shape

    # Compute the angle of the first principal component
    tensor_centered = tensors - torch.mean(tensors, dim=1, keepdim=True)
    U, S, V = torch.linalg.svd(tensor_centered, full_matrices=False)
    angles = torch.atan2(V[:, 0, 1], V[:, 0, 0])

    # Calculate and apply the desired rotation matrix
    # Note the signs, this matrix is post-multiplied
    angles -= (torch.pi / 4)
    rotation = torch.zeros((M, 2, 2), dtype=tensors.dtype, device=tensors.device)
    rotation[:, 0, 0] = torch.cos(angles)
    rotation[:, 0, 1] = -torch.sin(angles)
    rotation[:, 1, 0] = torch.sin(angles)
    rotation[:, 1, 1] = torch.cos(angles)

    rotated_coordinates = torch.bmm(tensors, rotation)

    return fit_unit_batch(rotated_coordinates)

def rotate_tensor_batch(tensors, angles):
    """
    Rotates a batch of coordinate tensors by the specified angles.

    Args:
    - tensors (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tensors
        in the batch and N is the number of points in each tensor.
    - angles (torch.Tensor): A 1D tensor of shape (M,) where M is the number of tensors in the batch.
    """
    M, N, _ = tensors.shape

    # Calculate and apply the desired rotation matrix
    # Note the signs, this matrix is post-multiplied
    rotation = torch.zeros((M, 2, 2), dtype=tensors.dtype, device=tensors.device)
    rotation[:, 0, 0] = torch.cos(angles)
    rotation[:, 0, 1] = -torch.sin(angles)
    rotation[:, 1, 0] = torch.sin(angles)
    rotation[:, 1, 1] = torch.cos(angles)

    return torch.bmm(tensors, rotation)

def translate_tensor_batch(tensors, x_diff, y_diff):
    """
    Translates a batch of coordinate tensors by the specified differences.

    Args:
    - tensors (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tensors
        in the batch and N is the number of points in each tensor.
    - x_diff (torch.Tensor): A 1D tensor of shape (M,) where M is the number of tensors in the batch.
    - y_diff (torch.Tensor): A 1D tensor of shape (M,) where M is the number of tensors in the batch.
    """
    M, N, _ = tensors.shape

    # Calculate and apply the desired translation matrix
    translated_coordinates = torch.zeros((M, N, 2), dtype=tensors.dtype, device=tensors.device)
    translated_coordinates[:, :, 0] = tensors[:, :, 0] + x_diff[:, None]
    translated_coordinates[:, :, 1] = tensors[:, :, 1] + y_diff[:, None]

    return translated_coordinates

def scale_tensor_batch(tensors, scale):
    """
    Scales a batch of coordinate tensors by the specified scale.

    Args:
    - tensors (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tensors
        in the batch and N is the number of points in each tensor.
    - scale (torch.Tensor): A 1D tensor of shape (M,) where M is the number of tensors in the batch.
    """
    M, N, _ = tensors.shape

    # Calculate and apply the desired scaling matrix
    scaled_coordinates = torch.zeros((M, 2, 2), dtype=tensors.dtype, device=tensors.device)
    scaled_coordinates = tensors * scale[:, None, None]

    return scaled_coordinates

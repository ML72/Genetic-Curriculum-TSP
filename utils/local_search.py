import torch


def tsp_length(positions, tour):
    """
    Calculates the length of a tour.

    Args:
    - positions (torch.Tensor): A 2D tensor of shape (N, 2).
    - tour (torch.Tensor): A 1D tensor of shape (N).
    """
    sorted_pos = positions[tour].float()

    # Don't forget to add distance from last point to first point
    dist = torch.linalg.norm(sorted_pos[:-1] - sorted_pos[1:], axis=-1).sum()
    dist += torch.linalg.norm(sorted_pos[-1] - sorted_pos[0], axis=-1)
    return dist

def tsp_length_batch(positions, tour):
    """
    Calculates the length of a tour.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2).
    - tour (torch.Tensor): A 2D tensor of shape (M, N).
    """
    sorted_pos = torch.zeros(positions.shape, dtype=positions.dtype, device=positions.device)
    for i in range(positions.shape[0]):
        sorted_pos[i] = positions[i][tour[i]]

    # Don't forget to add distance from last point to first point
    dist = torch.linalg.norm(sorted_pos[:,:-1,:] - sorted_pos[:,1:,:], axis=-1).sum(axis=1)
    dist += torch.linalg.norm(sorted_pos[:,-1,:] - sorted_pos[:,0,:], axis=-1)
    return dist

def _local_insert_batch(positions, tour, point):
    """
    Chooses and executes the position for local insertion, for a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2).
    - tour (torch.Tensor): A 2D tensor of shape (M, N).
    - point (torch.Tensor): Node to insert after, applied across the batch, between 0 inclusive and N exclusive.
    """
    # Safety check
    M, N = tour.shape
    device = positions.device
    assert point >= 0 and point < N

    # Precompute distances of edges
    idx = torch.arange(M, device=device)
    edge_dists = torch.zeros((M, N), device=device)
    for i in range(N):
        edge_dists[:,i] = torch.linalg.norm(
            positions[idx,tour[idx,i],:] - positions[idx,tour[idx,(i+1) % N],:], axis=-1
        )

    # Calculate costs
    costs = torch.zeros((M, N), device=device)
    e1_start = positions[idx,tour[idx,point],:]
    e1_end = positions[idx,tour[idx,(point+1) % N],:]
    for i in range(N):
        # Special cases, keep cost at 0
        if i == point or i == (point+1) % N:
            continue

        # General case
        i_pos = positions[idx,tour[idx,i],:]
        i_prev_pos = positions[idx,tour[idx,(i-1) % N],:]
        i_after_pos = positions[idx,tour[idx,(i+1) % N],:]
        costs[:,i] = torch.linalg.norm(i_prev_pos - i_after_pos, axis=-1) \
            + torch.linalg.norm(e1_start - i_pos, axis=-1) \
            + torch.linalg.norm(e1_end - i_pos, axis=-1) \
            - edge_dists[:,point] \
            - edge_dists[:,(i-1) % N] \
            - edge_dists[:,i]
    
    # Select best choice and insert if better than current option
    end = torch.argmin(costs, dim=1)
    tour_copy = tour.clone()
    for i in range(M):
        if costs[i,end[i]] < 0:
            if end[i] < point:
                tour[i, end[i]:point] = tour_copy[i, end[i]+1:point+1]
                tour[i, point] = tour_copy[i, end[i]]
            else:
                tour[i, point+2:end[i]+1] = tour_copy[i, point+1:end[i]]
                tour[i, point+1] = tour_copy[i, end[i]]

def _2opt_choice(positions, tour, e1, e2):
    """
    Chooses and executes the best 2-opt move between two edges.

    Args:
    - positions (torch.Tensor): A 2D tensor of shape (N, 2).
    - tour (torch.Tensor): A 1D tensor of shape (N).
    - e1 (Integer): Location of first edge start in tour, between 0 inclusive and N exclusive.
    - e2 (Integer): Location of second edge start in tour, between 0 inclusive and N exclusive.
    """
    # Ensure e1 < e2
    if e1 > e2:
        e1, e2 = e2, e1

    # Calculate ends of edges
    N = tour.shape[0]
    e1_end = (e1 + 1) % N
    e2_end = (e2 + 1) % N

    # Calculate old and new costs
    cost_old = torch.linalg.norm(positions[tour[e1]] - positions[tour[e1_end]], axis=-1) \
        + torch.linalg.norm(positions[tour[e2]] - positions[tour[e2_end]], axis=-1)
    cost_new = torch.linalg.norm(positions[tour[e1]] - positions[tour[e2]], axis=-1) \
        + torch.linalg.norm(positions[tour[e1_end]] - positions[tour[e2_end]], axis=-1)
    
    # Conditionally conduct 2-opt operation
    if cost_new < cost_old and e1 < e2-1:
        tour[e1_end:e2+1] = torch.flip(tour[e1_end:e2+1], [0])

def _2opt_choice_batch(positions, tour, e1, e2):
    """
    Chooses and executes the best 2-opt move between two edges, for a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2).
    - tour (torch.Tensor): A 2D tensor of shape (M, N).
    - e1 (torch.Tensor): Locations of first edge start in tour, of shape (M).
    - e2 (torch.Tensor): Locations of second edge start in tour, of shape (M).
    """
    # Ensure all e1 < e2
    mask = e1 > e2
    e1[mask], e2[mask] = e2[mask], e1[mask]

    # Calculate ends of edges
    M, N = tour.shape
    device = positions.device
    e1_end = (e1 + 1) % N
    e2_end = (e2 + 1) % N

    # Calculate old and new costs
    idx = torch.arange(M, device=device)
    cost_old = torch.linalg.norm(positions[idx,tour[idx,e1],:] - positions[idx,tour[idx,e1_end],:], axis=-1) \
        + torch.linalg.norm(positions[idx,tour[idx,e2],:] - positions[idx,tour[idx,e2_end],:], axis=-1)
    cost_new = torch.linalg.norm(positions[idx,tour[idx,e1],:] - positions[idx,tour[idx,e2],:], axis=-1) \
        + torch.linalg.norm(positions[idx,tour[idx,e1_end],:] - positions[idx,tour[idx,e2_end],:], axis=-1)
    
    # Conditionally conduct 2-opt operation
    # Not full vectorized because "only integer tensors of a single element can be converted to an index"
    mask = (cost_new < cost_old) & (e1 < e2-1)
    for i in range(M):
        if mask[i]:
            tour[i, e1_end[i]:e2[i]+1] = torch.flip(tour[i, e1_end[i]:e2[i]+1], [0])

def _2opt_search_batch(positions, tour, start):
    """
    Chooses and executes the best 2-opt move between two edges, for a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2).
    - tour (torch.Tensor): A 2D tensor of shape (M, N).
    - start (torch.Tensor): Starting edge, applied across the batch, between 0 inclusive and N-2 exclusive.
        Only goes to N-3 because swapping edge N-2 with N-1 is a no-op, with 0-based indexing.
    """
    # Safety check
    M, N = tour.shape
    device = positions.device
    assert start >= 0 and start < N-2

    # Calculate costs
    idx = torch.arange(M, device=device)
    costs = torch.zeros((M, N-start-2), device=device)
    e1_start = positions[idx,tour[idx,start],:]
    e1_end = positions[idx,tour[idx,start+1],:]
    e1_dist = torch.linalg.norm(e1_start - e1_end, axis=-1)
    for i in range(start+2, N):
        e2_start = positions[idx,tour[idx,i],:]
        e2_end = positions[idx,tour[idx,(i+1) % N],:]
        costs[:,i-start-2] = torch.linalg.norm(e1_start - e2_start, axis=-1) \
            + torch.linalg.norm(e1_end - e2_end, axis=-1) \
            - torch.linalg.norm(e2_start - e2_end, axis=-1) \
            - e1_dist
    
    # Special case, avoid consecutive edges
    if start == 0:
        costs[:,-1] = 0
    
    # Select best second edge and apply 2-opt if better than current option
    end = torch.argmin(costs, dim=1)
    for i in range(M):
        if costs[i,end[i]] < 0:
            tour[i, start+1:end[i]+start+3] = torch.flip(tour[i, start+1:end[i]+start+3], [0])

def local_insertion(positions, tours):
    """
    Conducts a local insertion heuristic on a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tours
        in the batch and N is the number of points in each tour.
    - tours (torch.Tensor): A 2D tensor of shape (M, N) where M is the number of tours and N
        is the number of points in each tour.
    """
    M, N = tours.shape
    for i in range(N):
        _local_insert_batch(positions, tours, i)

def random_2opt(positions, tours, alpha=0.5, beta=1.5):
    """
    Conducts random 2-opt on a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tours
        in the batch and N is the number of points in each tour.
    - tours (torch.Tensor): A 2D tensor of shape (M, N) where M is the number of tours and N
        is the number of points in each tour.
    - alpha (Float): Hyperparameter controlling strength
    - beta (Float): Hyperparameter controlling strength
    """
    M, N = tours.shape
    for i in range(int(alpha * (N ** beta))):
        e1 = torch.randint(0, N, (M,), device=positions.device)
        e2 = torch.randint(0, N, (M,), device=positions.device)
        _2opt_choice_batch(positions, tours, e1, e2)

def search_2opt(positions, tours):
    """
    Conducts search 2-opt on a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tours
        in the batch and N is the number of points in each tour.
    - tours (torch.Tensor): A 2D tensor of shape (M, N) where M is the number of tours and N
        is the number of points in each tour.
    """
    M, N = tours.shape
    for i in range(N-2):
        _2opt_search_batch(positions, tours, i)

def combined_local_search(positions, tour, iters=10, alpha=0.5, beta=1.5):
    """
    Conducts a combined local search on a batch of tours.

    Args:
    - positions (torch.Tensor): A 3D tensor of shape (M, N, 2) where M is the number of tours
        in the batch and N is the number of points in each tour.
    - tours (torch.Tensor): A 2D tensor of shape (M, N) where M is the number of tours and N
        is the number of points in each tour.
    - iters (Integer): Number of total iterations to conduct search
    - alpha (Float): Hyperparameter controlling strength for random 2-opt
    - beta (Float): Hyperparameter controlling strength for random 2-opt
    """
    for i in range(iters):
        local_insertion(positions, tour)
        random_2opt(positions, tour, alpha=alpha, beta=beta)
        search_2opt(positions, tour)

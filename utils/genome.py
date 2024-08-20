import torch


GENOME_SIZE = 6
MUTATE_PROB = 0.25 # 1/4*2/3=1/6, meaning 1 base will be mutated in expectation
GENOME_MAX = torch.tensor([25, 0.08, 2 * 3.1416, 1, 1, 1])
GENOME_MIN = torch.tensor([1, 0.03, -2 * 3.1416, 0.7, 0, 0])
GENOME_INC = torch.tensor([1, 0.01, 0.1, 0.1, 0.1, 0.1])

def sample_random_genomes(size):
    """
    Generates a random genome of the specified size.
    Each genome is described by 6 numbers:
    0. Link size: [1, 25]
    1. Link noise: [0.02, 0.1]
    2. Rotate angle: [-2*pi, 2*pi]
    3. Scale factor: [0.5, 1]
    4. Translate x: [0, 1]
    5. Translate y: [0, 1]

    Args:
    - size (int): The size of the genome.
    """
    genomes = torch.zeros((size, 6))
    genomes[:, 0] = torch.tensor([1, 5, 10, 15] * (size // 4))
    genomes[:, 1] = 0.05
    genomes[:, 3] = 1
    genomes[:, 4] = 0.5
    genomes[:, 5] = 0.5
    return genomes
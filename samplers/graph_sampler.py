import torch_geometric as PyG
from torch_geometric.data import Data


def node_sampling(G: Data):
    """
    Dictionary of Sampling methods. 3 in total:
    - SignalStrength: Sample nodes based on signal strength exhibited node weights + neighbors. Basically, sample nodes
    that influence output the most (our own).
    - Core: Core-base node sampling taken from FastGAE (https://arxiv.org/pdf/2002.01910.pdf)
    - Degree: Degree-based node sampling taken from FastGAE (https://arxiv.org/pdf/2002.01910.pdf)
    """
    pass

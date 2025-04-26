import torch

def batch_trace(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the trace of a batch of matrices.

    Args:
        matrix: A batch of matrices (batch_size, N, N)

    Returns:
        A tensor containing the trace for each matrix in the batch (batch_size,)
    """
    return torch.diagonal(matrix, dim1=1, dim2=2).sum(1)
import numpy as np
from numba import njit


@njit(cache=True)
def nb_mod_position_inplace(position, boundary):
    """In-place modulo for position array shaped as (dim, block_size)."""
    dim, block_size = position.shape
    for d in range(dim):
        position[d] = np.mod(position[d], boundary[d])

__doc__ = """ Rotation kernels """

import functools

import numpy as np
from numpy import sin, cos, sqrt, arccos
from numpy.typing import NDArray

from elastica._rotations import _get_rotation_matrix
from elastica._linalg import _batch_matvec

from numba import njit


@njit(cache=True)  # type: ignore
def _rotate_vector(
    vector_collection: NDArray[np.float64],
    scale: np.float64,
    axis_collection: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Rotate vector collection by specified axes and scale (alibi rotation).

    Parameters
    ----------
    vector_collection : numpy.ndarray
        2D array of shape (dim, blocksize) containing vectors to be rotated
    scale : float
        Scale factor for rotation angles. The actual rotation angle for each
        frame is scale * ||axis||, where ||axis|| is the magnitude of the
        corresponding axis vector.
    axis_collection : numpy.ndarray
        2D array of shape (dim, blocksize) containing rotation axes for each
        director frame. Each column represents the axis of rotation for the
        corresponding director frame.

    Returns
    -------
    vector_collection : numpy.ndarray
        2D array of shape (dim, blocksize) containing the rotated vectors
    """
    return _batch_matvec(
        _get_rotation_matrix(scale, axis_collection), vector_collection
    )

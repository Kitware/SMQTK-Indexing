from math import pi

import numpy as np
from scipy.spatial.distance import cdist


def histogram_intersection_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the histogram intersection distance between given histogram
    vectors or matrices ``a`` and ``b``, returning a value between ``0.0`` and
    ``1.0``. This is the inverse of intersection similarity, whereby a distance
    of  ``0.0`` means full intersection and ``1.0`` means no intersection.

    This implements non-branching formula for efficient computation.

    Input vectors ``a`` and ``b`` may be of 1 or 2 dimensions. Depending on the
    values of ``a`` and ``b``, different things may occur:

    * If both ``a`` and ``b`` are 2D matrices, they're shape must be congruent
      and the result will be an vector of distances between parallel rows.

    * If either is a 1D vector and the other is a 2D matrix, a vector of
      distances is returned between the 1D vector and each row of the 2D matrix.

    * If both ``a`` and ``b`` are 1D vectors, a floating-point scalar distance
      is returned that is the histogram intersection distance between the input
      vectors.

    :param a: Histogram or array of histograms ``a``
    :param b: Histogram or array of histograms ``b``

    :return: Float or array of float distance (inverse percent intersection).
    """
    # TODO: input value checks?
    # Which axis to sum on. If there is a matrix involved, its along column,
    # but if its just two arrays its along the row.
    #
    # - The following are noticeably slower:
    #       sum_axis = not (a.ndim == 1 and b.ndim == 1)
    #       sum_axis = (a.ndim > 1) | (b.ndim > 1)
    sum_axis = 1
    if a.ndim == 1 and b.ndim == 1:
        sum_axis = 0
    # TODO(john.moeller): Assuming each sums to 1, this can be sped up
    #   return np.abs(np.subtract(i, j)).sum(sum_axis) * 0.5
    return 1. - ((np.add(a, b) - np.abs(np.subtract(a, b))).sum(sum_axis) * 0.5)


def histogram_intersection_distance_fast(i: np.ndarray, j: np.ndarray) -> float:
    """
    Compute the histogram intersection percent relation between given 1D
    histogram vectors ``a`` and ``b``, returning a value between 0.0 and 1.0.
    0.0 means full intersection, and 1.0 means no intersection.

    This implements non-branching formula for efficient computation.

    Use of this implementations is faster when the input will only be 1D
    vectors.

    PENDING DEPRECATION: This function doesn't time much, if any, better than
        ``histogram_intersection_distance`` (via ipython %timeit using random
        input).

    :param i: Histogram ``i``
    :param j: Histogram ``j``

    :return: Float inverse percent intersection amount.

    """
    return 1.0 - ((i + j - np.abs(i - j)).sum() * 0.5)


def euclidean_distance(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Compute euclidean distance between two N-dimensional point vectors.

    :param i: Vector i
    :param j: Vector j

    :return: Float distance.

    """
    sum_axis = 1
    if i.ndim == 1 and j.ndim == 1:
        sum_axis = 0
    return np.sqrt(np.square(i - j).sum(sum_axis))


def cosine_similarity(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Angular similarity between vectors i and j. Results in a value between 1,
    where i and j are exactly the same, to -1, meaning exactly opposite. 0
    indicates orthogonality. Negative values will only be returned if input
    vectors can have negative values.

    See: http://en.wikipedia.org/wiki/Cosine_similarity

    :param i: Vector i
    :param j: Vector j

    :return: Float or array of cosine similarity.
    """
    # Only handling 1D-vectors for i
    assert i.ndim == 1

    i = i.reshape(1, -1)
    if j.ndim == 1:
        j = j.reshape(1, -1)

    # Cosine similarity function returns matrix in shape of (i_N_samples, j_N_samples)
    cosine_s = 1 - cdist(i, j, metric='cosine')[0]

    # Return a float if both i and j only have one sample
    if cosine_s.size == 1:
        return cosine_s[0]
    else:
        return cosine_s


def cosine_distance(i: np.ndarray, j: np.ndarray, pos_vectors: bool = True) -> np.ndarray:
    """
    Cosine similarity converted into angular distance.

    See: https://en.wikipedia.org/wiki/Cosine_similarity, section
    "Angular distance and similarity".

    :param i: Vector i
    :param j: Vector j
    :param pos_vectors: If we expect vector elements to always be positive.
        Default value is True (common case).

    :return: Float or array of cosine distance between [0, 1] range.
    """
    # limit to between -1 and 1
    sim = np.maximum(np.minimum(cosine_similarity(i, j), 1), -1)

    return (1 + bool(pos_vectors)) * np.arccos(sim) / pi


def hamming_distance(i: int, j: int) -> int:
    """
    Return the hamming distance between the two given pythonic integers, or the
    number of places where the bits differ.

    **Note:** *We say "pythonic" integer here because this function has no cap
    on the number of bits used to represent said integer. This function will
    execute correctly regardless whether i/j is 32 bits or 512 bits, etc."

    :param i: First integer.
    :param j: Second integer.

    :return: Integer hamming distance between the two values.
    """
    # TODO: Find something better than this?
    return bin(i ^ j).count('1')

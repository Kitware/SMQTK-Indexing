"""
Interface and plugin getter for LSH algorithm hash generation functors.
"""
import abc

import numpy as np

from smqtk_core import Configurable, Pluggable


class LshFunctor (Configurable, Pluggable):
    """
    Locality-sensitive hashing functor interface.

    The aim of such a function is to be able to generate hash codes
    (bit-vectors) such that similar items map to the same or similar hashes
    with a high probability. In other words, it aims to maximize hash collision
    for similar items.

    **Building Models**

    Some hash functions want to build a model based on some training set of
    descriptors. Due to the non-standard nature of algorithm training and model
    building, please refer to the specific implementation for further
    information on whether model training is needed and how it is accomplished.

    """

    def __call__(self, descriptor: np.ndarray) -> np.ndarray:
        return self.get_hash(descriptor)

    @abc.abstractmethod
    def get_hash(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Get the locality-sensitive hash code for the input descriptor.

        :param descriptor: Descriptor vector we should generate the hash of.

        :return: Generated bit-vector as a numpy array of booleans.

        """

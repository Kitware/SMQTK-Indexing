import abc
from typing import Iterable, Sequence, Tuple

import numpy as np

from smqtk_core import Configurable, Pluggable
from smqtk_indexing.utils.iter_validation import check_empty_iterable


class HashIndex (Configurable, Pluggable):
    """
    Specialized ``NearestNeighborsIndex`` for indexing unique hash codes
    bit-vectors) in memory (numpy arrays) using the hamming distance metric.

    Implementations of this interface cannot be used in place of something
    requiring a ``NearestNeighborsIndex`` implementation due to the speciality
    of this interface.

    Only unique bit vectors should be indexed. The ``nn`` method should not
    return the same bit vector more than once for any query.
    """

    def __len__(self) -> int:
        return self.count()

    @staticmethod
    def _empty_iterable_exception() -> BaseException:
        """
        Create the exception instance to be thrown when no descriptors are
        provided to ``build_index``/``update_index``.

        :return: ValueError instance to be thrown.
        :rtype: ValueError

        """
        return ValueError("No hash vectors in provided iterable.")

    def build_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Build the index with the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of hash vectors (boolean-valued) to build index
            over.

        """
        check_empty_iterable(hashes, self._build_index,
                             self._empty_iterable_exception())

    def update_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Additively update the current index with the one or more hash vectors
        given.

        If no index exists yet, a new one should be created using the given hash
        vectors.

        :raises ValueError: No data available in the given iterable.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.

        """
        check_empty_iterable(hashes, self._update_index,
                             self._empty_iterable_exception())

    def remove_from_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Partially remove hashes from this index.

        :param hashes: Iterable of numpy boolean hash vectors to remove from
            this index.

        :raises ValueError: No data available in the given iterable.
        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.

        """
        check_empty_iterable(hashes, self._remove_from_index,
                             self._empty_iterable_exception())

    def nn(self, h: np.ndarray, n: int = 1) -> Tuple[np.ndarray, Sequence[float]]:
        """
        Return the nearest `N` neighbor hash codes as bit-vectors to the given
        hash code bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        :raises ValueError: Current index is empty.

        :param h: Hash code vectors (boolean-valued) to compute the neighbors
            of. Should be the same bit length as indexed hash codes.
        :param n: Number of nearest neighbors to find.

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.

        """
        # Only check for count because we're no longer dealing with descriptor
        # elements.
        if not self.count():
            raise ValueError("No index currently set to query from!")
        return self._nn(h, n)

    @abc.abstractmethod
    def count(self) -> int:
        """
        :return: Number of elements in this index.
        """

    @abc.abstractmethod
    def _build_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Internal method to be implemented by sub-classes to build the index with
        the given hash codes (bit-vectors).

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :param hashes: Iterable of hash vectors (boolean-valued) to build index
            over.

        """

    @abc.abstractmethod
    def _update_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more hash vectors given.

        If no index exists yet, a new one should be created using the given hash
        vectors.

        :param hashes: Iterable of numpy boolean hash vectors to add to this
            index.

        """

    @abc.abstractmethod
    def _remove_from_index(self, hashes: Iterable[np.ndarray]) -> None:
        """
        Internal method to be implemented by sub-classes to partially remove
        hashes from this index.

        :param hashes: Iterable of numpy boolean hash vectors to remove from
            this index.

        :raises KeyError: One or more hashes provided do not match any stored
            hashes.  The index should not be modified.

        """

    @abc.abstractmethod
    def _nn(self, h: np.ndarray, n: int = 1) -> Tuple[np.ndarray, Tuple[float, ...]]:
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbor hash codes as bit-vectors to the given hash code
        bit-vector.

        Distances are in the range [0,1] and are the percent different each
        neighbor hash is from the query, based on the number of bits contained
        in the query (normalized hamming distance).

        When this internal method is called, we have already checked that our
        index is not empty.

        :param h: Hash code vector (boolean-valued) to compute the neighbors
            of. Should be the same bit length as indexed hash codes.
        :param n: Number of nearest neighbors to find.

        :return: Tuple of nearest N hash codes and a tuple of the distance
            values to those neighbors.

        """

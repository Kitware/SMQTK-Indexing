"""
This module contains a general locality-sensitive-hashing algorithm for
nearest neighbor indexing, and various implementations of LSH functors for use
in the base.
"""
import collections
import itertools
import logging
import multiprocessing
from typing import (
    cast,
    Any, Callable, Deque, Dict, Hashable, Iterable, Iterator, List, Optional,
    Set, Tuple, Type, TypeVar
)

import numpy

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict
)
from smqtk_core.dict import merge_dict
from smqtk_dataprovider import KeyValueStore
from smqtk_dataprovider.exceptions import ReadOnlyError
from smqtk_descriptors import DescriptorElement, DescriptorSet
from smqtk_descriptors.utils import parallel_map
from smqtk_indexing import HashIndex, LshFunctor, NearestNeighborsIndex
from smqtk_indexing.impls.hash_index.linear import LinearHashIndex
from smqtk_indexing.utils import metrics
from smqtk_indexing.utils.bits import bit_vector_to_int_large
from smqtk_indexing.utils.progress_reporter import ProgressReporter


LOG = logging.getLogger(__name__)
T_LSH = TypeVar("T_LSH", bound="LSHNearestNeighborIndex")


class LSHNearestNeighborIndex (NearestNeighborsIndex):
    """
    Locality-sensitive hashing based nearest neighbor index

    This type of algorithm relies on a hashing algorithm to hash descriptors
    such that similar descriptors are hashed the same or similar hash value.
    This allows simpler distance functions (hamming distance) to be performed
    on hashes in order to find nearby bins which are more likely to hold
    similar descriptors.

    LSH nearest neighbor algorithms consist of:
        * Index of descriptors to query over
        * A hashing function that transforms a descriptor vector into a
          hash (bit-vector).
        * Key-Value store of hash values to their set of hashed descriptor
          UUIDs.
        * Nearest neighbor index for indexing bit-vectors (treated as
          descriptors)

    """

    @classmethod
    def is_usable(cls) -> bool:
        # This "shell" class is always usable, no special dependencies.
        return True

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as
        arguments, turning those argument names into configuration dictionary
        keys. If any of those arguments have defaults, we will add those values
        into the configuration dictionary appropriately. The dictionary
        returned should only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this
        class.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        default = super(LSHNearestNeighborIndex, cls).get_default_config()

        lf_default = make_default_config(LshFunctor.get_impls())
        default['lsh_functor'] = lf_default

        di_default = make_default_config(DescriptorSet.get_impls())
        default['descriptor_set'] = di_default

        hi_default = make_default_config(HashIndex.get_impls())
        default['hash_index'] = hi_default

        h2u_default = make_default_config(KeyValueStore.get_impls())
        default['hash2uuids_kvstore'] = h2u_default

        return default

    @classmethod
    def from_config(
        cls: Type[T_LSH],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T_LSH:
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :type config_dict: dict

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.
        :type merge_default: bool

        :return: Constructed instance from the provided config.
        :rtype: LSHNearestNeighborIndex

        """
        # Controlling merge here so we can control known comment stripping from
        # default config.
        if merge_default:
            merged = cls.get_default_config()
            merge_dict(merged, config_dict)
        else:
            merged = config_dict

        merged['lsh_functor'] = \
            from_config_dict(merged['lsh_functor'], LshFunctor.get_impls())
        merged['descriptor_set'] = \
            from_config_dict(merged['descriptor_set'],
                             DescriptorSet.get_impls())

        # Hash index may be None for a default at-query-time linear indexing
        if merged['hash_index'] and merged['hash_index']['type']:
            merged['hash_index'] = \
                from_config_dict(merged['hash_index'],
                                 HashIndex.get_impls())
        else:
            LOG.debug("No HashIndex impl given. Passing ``None``.")
            merged['hash_index'] = None

        # remove possible comment added by default generator
        if 'hash_index_comment' in merged:
            del merged['hash_index_comment']

        merged['hash2uuids_kvstore'] = \
            from_config_dict(merged['hash2uuids_kvstore'],
                             KeyValueStore.get_impls())

        return super(LSHNearestNeighborIndex, cls).from_config(merged, False)

    def __init__(
        self,
        lsh_functor: LshFunctor,
        descriptor_set: DescriptorSet,
        hash2uuids_kvstore: KeyValueStore,
        hash_index: Optional[HashIndex] = None,
        distance_method: str = 'cosine',
        read_only: bool = False
    ):
        """
        Initialize LSH algorithm with a hashing functor, descriptor index and
        hash nearest-neighbor index.

        In order to provide out-of-the-box neighbor querying ability, at least
        the ``descriptor_set`` and ``hash2uuids_kvstore`` must be provided.
        The UIDs of descriptors in the ``descriptor_set`` should be fully
        mapped by the key-value mapping (``hash2uuids_kvstore``). If not, not
        all descriptors will be accessible via the nearest-neighbor query (not
        referenced in ``hash2uuids_kvstore`` map), or the requested number of
        neighbors might not be returned (descriptors hashed in ``hash_index``
        disjoint from ``descriptor_set``).

        An ``LSHNearestNeighborIndex`` instance is effectively read-only if any
        of its input structures (`descriptor_set`, `hash2uuids_kvstore`,
        `hash_index`) are read-only.

        :param lsh_functor: LSH functor implementation instance.
        :param descriptor_set: Index in which DescriptorElements will be
            stored.
        :param hash2uuids_kvstore: KeyValueStore instance to use for linking a
            hash code, as an integer, in the ``hash_index`` with one or more
            ``DescriptorElement`` instance UUIDs in the given
            ``descriptor_set``.
        :param hash_index: ``HashIndex`` for indexing unique hash codes using
            hamming distance.

            If this is set to ``None`` (default), we will perform brute-force
            linear neighbor search for each query based on the hash codes
            currently in the hash2uuid index using hamming distance
        :param distance_method: String label of distance method to use for
            determining descriptor similarity (after finding near hashes for a
            given query).

            This must one of the following:
                - "euclidean": Simple euclidean distance between two
                    descriptors (L2 norm).
                - "cosine": Cosine angle distance/similarity between two
                    descriptors.
                - "hik": Histogram intersection distance between two
                    descriptors.
        :param read_only: If this index should only read from its configured
            descriptor and hash indexes. This will cause a ``ReadOnlyError`` to
            be raised from build_index.

        :raises ValueError: Invalid distance method specified.

        """
        super(LSHNearestNeighborIndex, self).__init__()

        # TODO(paul.tunison): Add in-memory empty defaults for
        #   descriptor_set/hash2uuids_kvstore attributes.
        self.lsh_functor = lsh_functor
        self.descriptor_set = descriptor_set
        self.hash_index = hash_index
        # Will use with int|long keys and set[collection.Hashable] values.
        self.hash2uuids_kvstore = hash2uuids_kvstore
        self.distance_method = distance_method
        self.read_only = read_only

        # Lock for model component access (combination of descriptor-set,
        # hash_index and kvstore).  Multiprocessing because resources can be
        # potentially modified on other processes.
        self._model_lock = multiprocessing.RLock()

        self._distance_function = self._get_dist_func(self.distance_method)

    @staticmethod
    def _get_dist_func(
        distance_method: str
    ) -> Callable[[numpy.ndarray, numpy.ndarray], float]:
        """
        Return appropriate distance function given a string label.

        :raises ValueError: Unrecognized distance method identifier is passed.
        """
        if distance_method == "euclidean":
            return metrics.euclidean_distance
        elif distance_method == "cosine":
            # Inverse of cosine similarity function return
            return metrics.cosine_distance
        elif distance_method == 'hik':
            return metrics.histogram_intersection_distance_fast
        else:
            # TODO: Support scipy/scikit-learn distance methods
            raise ValueError("Invalid distance method label. Must be one of "
                             "['euclidean' | 'cosine' | 'hik']")

    def get_config(self) -> Dict[str, Any]:
        hi_conf = None
        if self.hash_index is not None:
            hi_conf = to_config_dict(self.hash_index)
        return {
            "lsh_functor": to_config_dict(self.lsh_functor),
            "descriptor_set": to_config_dict(self.descriptor_set),
            "hash_index": hi_conf,
            "hash2uuids_kvstore":
                to_config_dict(self.hash2uuids_kvstore),
            "distance_method": self.distance_method,
            "read_only": self.read_only,
        }

    def count(self) -> int:
        """
        :return: Maximum number of descriptors reference-able via a
            nearest-neighbor query (count of descriptor index). Actual return
            may be smaller of hash2uuids mapping is not complete.
        """
        with self._model_lock:
            c = 0
            for set_v in self.hash2uuids_kvstore.values():
                c += len(set_v)
            return c

    def _build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Internal method to be implemented by sub-classes to build the index
        with the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        :raises ReadOnlyError: This index is set to be read-only and cannot be
            modified.

        :param descriptors: Iterable of descriptor elements to build index
            over.

        """
        with self._model_lock:
            if self.read_only:
                raise ReadOnlyError("Cannot modify container attributes due "
                                    "to being in read-only mode.")

            LOG.debug("Clearing and adding new descriptor elements")
            self.descriptor_set.clear()
            self.descriptor_set.add_many_descriptors(descriptors)

            LOG.debug("Generating hash codes")
            hash_vectors: Deque[numpy.ndarray] = collections.deque()
            self.hash2uuids_kvstore.clear()
            prog_reporter = ProgressReporter(LOG.debug, 1.0).start()
            # We just cleared the previous store, so aggregate new kv-mapping
            # in ``kvstore_update`` for single update after loop.
            # NOTE: Mapping type apparently not yet covariant in the key type.
            kvstore_update: Dict[Hashable, Set[Hashable]] = collections.defaultdict(set)
            for d in self.descriptor_set:
                h_vec = self.lsh_functor.get_hash(d.vector())
                hash_vectors.append(h_vec)
                h_int = bit_vector_to_int_large(h_vec)
                kvstore_update[h_int] |= {d.uuid()}
                prog_reporter.increment_report()
            prog_reporter.report()
            self.hash2uuids_kvstore.add_many(kvstore_update)
            del kvstore_update

            if self.hash_index is not None:
                LOG.debug(f"Clearing and building hash index of type {type(self.hash_index)}")
                # a build is supposed to clear previous state.
                self.hash_index.build_index(hash_vectors)

    def _update_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more descriptor elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        :raises ReadOnlyError: This index is set to be read-only and cannot be
            modified.

        :param descriptors: Iterable of descriptor elements to add to this
            index.

        """
        with self._model_lock:
            if self.read_only:
                raise ReadOnlyError("Cannot modify container attributes due "
                                    "to being in read-only mode.")
            # tee out iterable for use in adding to index as well as hash code
            # generation.
            d_for_index, d_for_hashing = itertools.tee(descriptors, 2)

            LOG.debug("Updating descriptor index.")
            self.descriptor_set.add_many_descriptors(d_for_index)

            LOG.debug("Generating hash codes for new descriptors")
            prog_reporter = ProgressReporter(LOG.debug, 1.0).start()
            # for updating hash_index
            hash_vectors: Deque[numpy.ndarray] = collections.deque()
            # for updating kv-store after collecting new hash codes
            # NOTE: Mapping type apparently not yet covariant in the key type.
            kvstore_update: Dict[Hashable, Set[Hashable]] = {}
            for d in d_for_hashing:
                h_vec = self.lsh_functor.get_hash(d.vector())
                hash_vectors.append(h_vec)
                h_int = bit_vector_to_int_large(h_vec)
                # Get, update and reinsert hash UUID set object.
                if h_int not in kvstore_update:
                    #: :type: set
                    kvstore_update[h_int] = \
                        self.hash2uuids_kvstore.get(h_int, set())
                kvstore_update[h_int] |= {d.uuid()}
                prog_reporter.increment_report()
            prog_reporter.report()

            LOG.debug("Updating kv-store with new hash codes")
            self.hash2uuids_kvstore.add_many(kvstore_update)
            del kvstore_update

            if self.hash_index is not None:
                LOG.debug("Updating hash index structure.")
                self.hash_index.update_index(hash_vectors)

    def _remove_from_index(self, uids: Iterable[Hashable]) -> None:
        """
        Remove descriptors from this index associated with the given UIDs.

        :param uids: Iterable of UIDs of descriptors to remove from this index.

        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.  The index should not be modified.
        :raises ReadOnlyError: This index is set to be read-only and cannot be
            modified.

        """
        with self._model_lock:
            if self.read_only:
                raise ReadOnlyError("Cannot modify container attributes due "
                                    "to being in read-only mode.")

            uids = list(uids)

            # Remove UIDs from our hash2uid-kvs
            # - get the hash for each input UID's descriptor, remove UID from
            #   recorded association set.
            # - `get_many_descriptors` fails when bad UIDs are provided
            #   (KeyError).
            LOG.debug("Removing hash2uid entries for UID's descriptors")
            h_vectors: Deque[numpy.ndarray] = collections.deque()
            h_ints: Deque[int] = collections.deque()
            for d in self.descriptor_set.get_many_descriptors(uids):
                h_vec = self.lsh_functor.get_hash(d.vector())
                h_vectors.append(h_vec)
                h_int = bit_vector_to_int_large(h_vec)
                h_ints.append(h_int)

            # If we're here, then all given UIDs mapped to an indexed
            # descriptor.  Proceed with removal from hash2uids kvs.  If a hash
            # no longer maps anything, remove that key from the KVS.
            hashes_for_removal: Deque[numpy.ndarray] = collections.deque()
            # store key-value pairs to update after loop in batch call
            # NOTE: Mapping type apparently not yet covariant in the key type.
            kvs_update: Dict[Hashable, Set[Hashable]] = {}
            # store keys to remove after loop in batch-call
            kvs_remove = set()
            for uid, h_int, h_vec in zip(uids, h_ints, h_vectors):
                if h_int not in kvs_update:
                    # First time seeing key, cache current value
                    kvs_update[h_int] = \
                        self.hash2uuids_kvstore.get(h_int, set())
                kvs_update[h_int] -= {uid}
                # If the resolves UID set is empty, flag the key for removal.
                if not kvs_update[h_int]:
                    del kvs_update[h_int]
                    kvs_remove.add(h_int)
                    hashes_for_removal.append(h_vec)
            LOG.debug("Updating hash2uuids: modified relations")
            self.hash2uuids_kvstore.add_many(kvs_update)
            LOG.debug("Updating hash2uuids: removing empty hash keys")
            self.hash2uuids_kvstore.remove_many(kvs_remove)
            del kvs_update, kvs_remove

            # call remove-from-index on hash-index if we have one and there are
            # hashes to be removed.
            if self.hash_index and hashes_for_removal:
                self.hash_index.remove_from_index(hashes_for_removal)

            # Remove descriptors from our set matching the given UIDs.
            self.descriptor_set.remove_many_descriptors(uids)

    def _nn(
        self,
        d: DescriptorElement,
        n: int = 1
    ) -> Tuple[Tuple[DescriptorElement, ...], Tuple[float, ...]]:
        """
        Internal method to be implemented by sub-classes to return the nearest
        `N` neighbors to the given descriptor element.

        When this internal method is called, we have already checked that there
        is a vector in ``d`` and our index is not empty.

        :param d: Descriptor element to compute the neighbors of.
        :param n: Number of nearest neighbors to find.

        :return: Tuple of nearest N DescriptorElement instances, and a tuple of
            the distance values to those neighbors.

        """
        LOG.debug("generating hash for descriptor")
        d_v = d.vector()
        d_h = self.lsh_functor.get_hash(d_v)

        def comp_descr_dist(d2_v: numpy.ndarray) -> float:
            return self._distance_function(d_v, d2_v)

        with self._model_lock:
            LOG.debug("getting near hashes")
            hi = self.hash_index
            if hi is None:
                # Make on-the-fly linear index
                hi = LinearHashIndex()
                # not calling ``build_index`` because we already have the int
                # hashes.
                hi.index = set(cast(Iterator[int], self.hash2uuids_kvstore.keys()))
            near_hashes, _ = hi.nn(d_h, n)

            LOG.debug("getting UUIDs of descriptors for nearby hashes")
            neighbor_uuids: List[Hashable] = []
            for h_int in map(bit_vector_to_int_large, near_hashes):
                # If descriptor hash not in our map, we effectively skip it.
                # Get set of descriptor UUIDs for a hash code.
                near_uuids: Set[Hashable] = self.hash2uuids_kvstore.get(h_int, set())
                # Accumulate matching descriptor UUIDs to a list.
                neighbor_uuids.extend(near_uuids)
            LOG.debug("-- matched %d UUIDs", len(neighbor_uuids))

            LOG.debug("getting descriptors for neighbor_uuids")
            neighbors = \
                list(self.descriptor_set.get_many_descriptors(neighbor_uuids))

        # Done with model parts at this point, so releasing lock.

        LOG.debug(f"ordering descriptors via distance method {self.distance_method}")
        LOG.debug('-- getting element vectors')
        neighbor_vectors = numpy.asarray(list(
            parallel_map(lambda d_: d_.vector(), neighbors)
        ))
        LOG.debug('-- calculating distances')
        distances = list(map(comp_descr_dist, neighbor_vectors))
        LOG.debug('-- ordering')
        ordered = sorted(zip(neighbors, distances),
                         key=lambda p: p[1])
        LOG.debug(f'-- slicing top n={n}')
        r_descrs: Tuple[DescriptorElement, ...]
        r_dists: Tuple[float, ...]
        r_descrs, r_dists = zip(*(ordered[:n]))
        return r_descrs, r_dists

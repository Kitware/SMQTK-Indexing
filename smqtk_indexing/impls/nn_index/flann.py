import itertools
import logging
import multiprocessing
import os
import pickle
import tempfile
from typing import cast, Any, Dict, Hashable, Iterable, List, Optional, Tuple
import warnings

import numpy

from smqtk_dataprovider import from_uri, DataElement
from smqtk_descriptors import DescriptorElement
from smqtk_descriptors.utils import parallel_map
from smqtk_indexing import NearestNeighborsIndex

# Requires FLANN bindings
try:
    import pyflann  # type: ignore
except ImportError:
    pyflann = None


LOG = logging.getLogger(__name__)


class FlannNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest-neighbor computation using the FLANN library (pyflann module).

    This implementation uses in-memory data structures, and thus has an index
    size limit based on how much memory the running machine has available.

    NOTE ON MULTIPROCESSING
        Normally, FLANN indices don't play well when multiprocessing due to
        the underlying index being a C structure, which doesn't auto-magically
        transfer to forked processes like python structure data does. The
        serialized FLANN index file is used to restore a built index in
        separate processes, assuming one has been built.

    """

    @classmethod
    def is_usable(cls) -> bool:
        # if underlying library is not found, the import above will error
        return pyflann is not None

    def __init__(
        self,
        index_uri: Optional[str] = None,
        parameters_uri: Optional[str] = None,
        descriptor_cache_uri: Optional[str] = None,
        # Parameters for building an index
        autotune: bool = False,
        target_precision: float = 0.95,
        sample_fraction: float = 0.1,
        distance_method: str = 'hik',
        random_seed: Optional[int] = None
    ):
        """
        Initialize FLANN index properties. Does not contain a query-able index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        When using this algorithm in a multiprocessing environment, the model
        file path parameters must be specified due to needing to reload the
        FLANN index on separate processes. This is because FLANN is in C and
        its instances are not copied into processes.

        Documentation on index building parameters and their meaning can be
        found in the FLANN documentation PDF:

            http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf

        See the MATLAB section for detailed descriptions (python section will
        just point you to the MATLAB section).

        :param index_uri: Optional URI to where to load/store FLANN index
            when initialized and/or built. If not configured, no model files
            are written to or loaded from disk.
        :param parameters_uri: Optional file location to load/save FLANN
            index parameters determined at build time. If not configured, no
            model files are written to or loaded from disk.
        :param descriptor_cache_uri: Optional file location to load/store
            DescriptorElements in this index. If not configured, no model files
            are written to or loaded from disk.
        :param autotune: Whether or not to perform parameter auto-tuning when
            building the index. If this is False, then the `target_precision`
            and `sample_fraction` parameters are not used.
        :param target_precision: Target estimation accuracy when determining
            nearest neighbor when tuning parameters. This should be between
            [0,1] and represents percentage accuracy.
        :param sample_fraction: Sub-sample percentage of the total index to use
            when performing auto-tuning. Value should be in the range of [0,1]
            and represents percentage.
        :param distance_method: Method label of the distance function to use.
            See FLANN documentation manual for available methods. Common
            methods include "hik", "chi_square" (default), and "euclidean".
            When loading and existing index, this value is ignored in
            preference for the distance method used to build the loaded index.
        :param random_seed: Integer to use as the random number generator seed.

        """
        warnings.warn(
            "This FLANN implementation is deprecated. Please utilize a more "
            "recent and supported plugin NearestNeighborsIndex, like the "
            "FaissNearestNeighborsIndex plugin.",
            category=DeprecationWarning
        )

        super(FlannNearestNeighborsIndex, self).__init__()

        self._index_uri = index_uri
        self._index_param_uri = parameters_uri
        self._descr_cache_uri = descriptor_cache_uri

        # Elements will be None if input URI is None
        self._index_elem = cast(
            Optional[DataElement],
            self._index_uri and from_uri(self._index_uri)
        )
        self._index_param_elem = cast(
            Optional[DataElement],
            self._index_param_uri and from_uri(self._index_param_uri)
        )
        self._descr_cache_elem = cast(
            Optional[DataElement],
            self._descr_cache_uri and from_uri(self._descr_cache_uri)
        )

        # parameters for building an index
        self._build_autotune = autotune
        self._build_target_precision = float(target_precision)
        self._build_sample_frac = float(sample_fraction)
        self._distance_method = str(distance_method)

        # Lock for model component access.  Using a multiprocessing due to
        # possible cases where another thread/process attempts to restore a
        # model before its fully written.  A reordering of _build_index could
        # lessen the requirement to a `threading.RLock`.
        self._model_lock = multiprocessing.RLock()

        # In-order cache of descriptors we're indexing over.
        # - flann.nn_index will spit out indices to list
        self._descr_cache: List[DescriptorElement] = []

        # The flann instance with a built index. None before index load/build.
        self._flann: pyflann.index.FLANN = None
        # Flann index parameters determined during building. None before index
        # load/build.
        self._flann_build_params = None

        #: :type: None | int
        self._rand_seed = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # The process ID that the currently set FLANN instance was built/loaded
        # on. If this differs from the current process ID, the index should be
        # reloaded from cache.
        self._pid: Optional[int] = None

        # Load the index/parameters if one exists
        if self._has_model_data():
            LOG.info("Found existing model data. Loading.")
            self._load_flann_model()

    def get_config(self) -> Dict[str, Any]:
        return {
            "index_uri": self._index_uri,
            "parameters_uri": self._index_param_uri,
            "descriptor_cache_uri": self._descr_cache_uri,
            "autotune": self._build_autotune,
            "target_precision": self._build_target_precision,
            "sample_fraction": self._build_sample_frac,
            "distance_method": self._distance_method,
            "random_seed": self._rand_seed,
        }

    def _has_model_data(self) -> bool:
        """
        check if configured model files are configured and not empty
        """
        return bool(self._index_elem and not self._index_elem.is_empty() and
                    self._index_param_elem and
                    not self._index_param_elem.is_empty() and
                    self._descr_cache_elem and
                    not self._descr_cache_elem.is_empty())

    def _load_flann_model(self) -> None:
        """
        Load an existing FLANN model from the current cache references.

        This sets the `_pid` attribute because the loaded model is only valid
        for the current process (C-library backed resources, not python).

        :raises ValueError: One or more descriptor elements in the cache does
            not have a retrievable vector.
        """
        if (
            bool(self._descr_cache) and
            self._descr_cache_elem is not None and
            not self._descr_cache_elem.is_empty()
        ):
            # Load descriptor cache
            # - is copied on fork, so only need to load here.
            LOG.debug("Loading cached descriptors")
            self._descr_cache = pickle.loads(
                self._descr_cache_elem.get_bytes()
            )

        # Params pickle include the build params + our local state params
        if self._index_param_elem and not self._index_param_elem.is_empty():
            state = pickle.loads(self._index_param_elem.get_bytes())
            self._build_autotune = state['b_autotune']
            self._build_target_precision = state['b_target_precision']
            self._build_sample_frac = state['b_sample_frac']
            self._distance_method = state['distance_method']
            self._flann_build_params = state['flann_build_params']

        # Load the binary index
        if (
            self._index_elem
            and not self._index_elem.is_empty()
        ):
            # make numpy matrix of descriptor vectors for FLANN
            d_vec_list = [d.vector() for d in self._descr_cache]
            if None in d_vec_list:
                raise ValueError(
                    "One or more descriptor elements do not have retrievable "
                    "vector data (returned None)."
                )
            pts_array = numpy.array(
                d_vec_list,
                dtype=cast(numpy.ndarray, d_vec_list[0]).dtype
            )
            pyflann.set_distance_type(self._distance_method)
            self._flann = pyflann.FLANN()
            tmp_fp = self._index_elem.write_temp()
            self._flann.load_index(tmp_fp, pts_array)
            self._index_elem.clean_temp()
            del pts_array, tmp_fp

        # Set current PID to the current
        self._pid = multiprocessing.current_process().pid

    def _restore_index(self) -> None:
        """
        If we think we're suppose to have an index, check the recorded PID with
        the current PID, reloading the index from cache if they differ.

        If there is a loaded index and we're on the same process that created
        it this does nothing.
        """
        if bool(self._flann) \
                and self._has_model_data() \
                and self._pid != multiprocessing.current_process().pid:
            self._load_flann_model()

    def count(self) -> int:
        """
        :return: Number of elements in this index.
        :rtype: int
        """
        with self._model_lock:
            if self._descr_cache is not None:
                return len(self._descr_cache)
            else:
                return 0

    def _build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Internal method to be implemented by sub-classes to build the index
        with the given descriptor data elements.

        Subsequent calls to this method should rebuild the current index.  This
        method shall not add to the existing index nor raise an exception to as
        to protect the current index.

        Implementation Notes:
            - We keep a cache file serialization around for our index in case
                sub-processing occurs so as to be able to recover from the
                underlying C data not being there. This could cause issues if
                a main or child process rebuild's the index, as we clear the
                old cache away.

        :param descriptors: Iterable of descriptor elements to build index
            over.
        :type descriptors:
            collections.abc.Iterable[smqtk.representation.DescriptorElement]

        """
        with self._model_lock:
            # Not caring about restoring the index because we're just making a
            # new one.
            LOG.info("Building new FLANN index")

            LOG.debug("Caching descriptor elements")
            self._descr_cache = list(descriptors)
            # Cache descriptors if we have an element
            if self._descr_cache_elem and self._descr_cache_elem.writable():
                LOG.debug(f"Caching descriptors: {self._descr_cache_elem}")
                self._descr_cache_elem.set_bytes(
                    pickle.dumps(self._descr_cache, -1)
                )

            params = {
                "target_precision": self._build_target_precision,
                "sample_fraction": self._build_sample_frac,
                "log_level": ("info"
                              if LOG.getEffectiveLevel() <= logging.DEBUG
                              else "warning")
            }
            if self._build_autotune:
                params['algorithm'] = "autotuned"
            if self._rand_seed is not None:
                params['random_seed'] = self._rand_seed
            pyflann.set_distance_type(self._distance_method)

            LOG.debug("Accumulating descriptor vectors into matrix for FLANN")
            pts_array = numpy.asarray(list(
                parallel_map(lambda d_: d_.vector(), self._descr_cache)
            ))

            LOG.debug('Building FLANN index')
            self._flann = pyflann.FLANN()
            self._flann_build_params = self._flann.build_index(pts_array,
                                                               **params)
            del pts_array

            if self._index_elem and self._index_elem.writable():
                LOG.debug("Caching index: %s", self._index_elem)
                # FLANN wants to write to a file, so make a temp file, then
                # read it in, putting bytes into element.
                fd, fp = tempfile.mkstemp()
                try:
                    self._flann.save_index(fp)
                    # Use the file descriptor to create the file object.
                    # This avoids reopening the file and will automatically
                    # close the file descriptor on exiting the with block.
                    # fdopen() is required because in Python 2 open() does
                    # not accept a file descriptor.
                    with os.fdopen(fd, 'rb') as f:
                        self._index_elem.set_bytes(f.read())
                finally:
                    os.remove(fp)
            if self._index_param_elem and self._index_param_elem.writable():
                LOG.debug(f"Caching index params: {self._index_param_elem}")
                state = {
                    'b_autotune': self._build_autotune,
                    'b_target_precision': self._build_target_precision,
                    'b_sample_frac': self._build_sample_frac,
                    'distance_method': self._distance_method,
                    'flann_build_params': self._flann_build_params,
                }
                self._index_param_elem.set_bytes(pickle.dumps(state, -1))

            self._pid = multiprocessing.current_process().pid

    def _update_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Internal method to be implemented by sub-classes to additively update
        the current index with the one or more descriptor elements given.

        If no index exists yet, a new one should be created using the given
        descriptors.

        The currently bundled FLANN implementation bindings (v1.8.4) does not
        support support incremental updating of an existing index.  Thus this
        update method fully rebuilds the index based on the previous cache of
        descriptors and the newly specified ones.  Due to requiring a full
        rebuild this update method may take a significant amount of time
        depending on the size of the index being updated.

        :param descriptors: Iterable of descriptor elements to add to this
            index.

        """
        with self._model_lock:
            self._restore_index()
            # Build a new index that contains the union of the current
            # descriptors and the new provided descriptors.
            LOG.info("Rebuilding FLANN index to include new descriptors.")
            self.build_index(itertools.chain(self._descr_cache, descriptors))

    def _remove_from_index(self, uids: Iterable[Hashable]) -> None:
        """
        Internal method to be implemented by sub-classes to partially remove
        descriptors from this index associated with the given UIDs.

        :param uids: Iterable of UIDs of descriptors to remove from this index.
        :type uids: collections.abc.Iterable[collections.abc.Hashable]

        :raises KeyError: One or more UIDs provided do not match any stored
            descriptors.

        """
        with self._model_lock:
            self._restore_index()
            uidset = set(uids)
            # Make sure provided UIDs are part of our current index.
            uid_diff = uidset - set(map(lambda d: d.uuid(), self._descr_cache))
            if uid_diff:
                if len(uid_diff) == 1:
                    raise KeyError(list(uid_diff)[0])
                else:
                    raise KeyError(uid_diff)
            # Filter descriptors NOT matching UIDs of current descriptor
            # cache.
            self._descr_cache = \
                [descr for descr in self._descr_cache if descr.uuid() in uidset]
            self.build_index(self._descr_cache)

    def _nn(
        self, d: DescriptorElement,
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
        with self._model_lock:
            self._restore_index()
            assert self._flann is not None, (
                "We should have an index after restoration."
            )

            vec = d.vector()

            # If the distance method is HIK, we need to treat it special since
            # that method produces a similarity score, not a distance score.
            #
            # FLANN asserts that we query for <= index size, thus the use of
            # min().
            idxs: numpy.ndarray
            dists: numpy.ndarray
            if self._distance_method == 'hik':
                # This call is different than the else version in that k is the
                # size of the full data set, so that we can reverse the
                # distances.
                idxs, dists = self._flann.nn_index(
                    vec, len(self._descr_cache),
                    **self._flann_build_params
                )
            else:
                idxs, dists = self._flann.nn_index(
                    vec, min(n, len(self._descr_cache)),
                    **self._flann_build_params
                )

            # When N>1, return value is a 2D array. Since this method limits
            # query to a single descriptor, we reduce to 1D arrays.
            if len(idxs.shape) > 1:
                idxs = idxs[0]
                dists = dists[0]

            if self._distance_method == 'hik':
                # Invert values to stay consistent with other distance value
                # norms. This also means that we reverse the "nearest" order
                # and reintroduce `n` size limit.
                # - This is intentionally happening *after* the "squeeze" op
                #   above.
                dists = (1.0 - dists)[::-1][:n]
                idxs = idxs[::-1][:n]

            return tuple(self._descr_cache[i] for i in idxs), tuple(dists)


NN_INDEX_CLASS = FlannNearestNeighborsIndex

from dataclasses import dataclass
from itertools import chain, groupby
import logging
import pickle
from os import path as osp
import threading
from typing import (
    cast, Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple,
    Type, TypeVar
)

import numpy as np

from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict
)
from smqtk_core.dict import merge_dict
from smqtk_dataprovider.exceptions import ReadOnlyError
from smqtk_dataprovider.utils.file import safe_create_dir
from smqtk_descriptors import DescriptorElement, DescriptorSet
from smqtk_descriptors.utils import parallel_map
from smqtk_indexing import NearestNeighborsIndex


CHUNK_SIZE = 5000
LOG = logging.getLogger(__name__)
T = TypeVar("T", bound="MRPTNearestNeighborsIndex")


@dataclass
class TreeElement:
    # (n_feats, depth) shaped float array
    random_basis: np.ndarray
    # 1D float array of size based on depth (see L305) TODO: Update line #
    splits: np.ndarray
    # Descriptor UIDs belonging to tree leaves.
    leaves: List[List[Hashable]]


class MRPTNearestNeighborsIndex (NearestNeighborsIndex):
    """
    Nearest Neighbors index that uses the MRPT algorithm of [Hyvönen et
    al](https://arxiv.org/abs/1509.06957).

    Multiple Random Projection Trees (MRPT) combines multiple shallow binary
    trees of a set depth to quickly search for near neighbors. Each tree has a
    separate set of random projections used to divide the dataset into a tree
    structure. This algorithm differs from most RP tree implementations in
    that all separators at a particular level in the tree share the same
    projection, to save on space. Every branch partitions a set of points into
    two equal portions relative to the corresponding random projection.

    On query, the leaf corresponding to the query vector is found in each
    tree. The neighbors are drawn from the set of points that are in the most
    leaves.

    The performance will depend on settings for the parameters:

    - If `depth` is too high, then the leaves will not have enough points
        to satisfy a query, and num_trees will need to be higher in order to
        compensate. If `depth` is too low, then performance may suffer because
        the leaves are large. If `N` is the size of the dataset, and `L =
        N/2^{depth}`, then leaves should be small enough that all
        `num_trees*L` descriptors that result from a query will fit easily in
        cache. Since query complexity is linear in `depth`, this parameter
        should be kept as low as possible.
    - The `num_trees` parameter will lower the variance of the results for
        higher values, but at the cost of using more memory on any particular
        query. As a rule of thumb, for a given value of `k`, num_trees should
        be about `3k/L`.
    """

    @classmethod
    def is_usable(cls) -> bool:
        return True

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        default = super(MRPTNearestNeighborsIndex, cls).get_default_config()

        di_default = make_default_config(DescriptorSet.get_impls())
        default['descriptor_set'] = di_default

        return default

    @classmethod
    def from_config(
        cls: Type[T],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T:
        if merge_default:
            cfg = cls.get_default_config()
            merge_dict(cfg, config_dict)
        else:
            cfg = config_dict

        cfg['descriptor_set'] = \
            from_config_dict(cfg['descriptor_set'],
                             DescriptorSet.get_impls())

        return super(MRPTNearestNeighborsIndex, cls).from_config(cfg, False)

    def __init__(
        self,
        descriptor_set: DescriptorSet,
        index_filepath: Optional[str] = None,
        parameters_filepath: Optional[str] = None,
        read_only: bool = False,
        # Parameters for building an index
        num_trees: int = 10,
        depth: int = 1,
        random_seed: Optional[int] = None,
        pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
        use_multiprocessing: bool = False
    ):
        """
        Initialize MRPT index properties. Does not contain a queryable index
        until one is built via the ``build_index`` method, or loaded from
        existing model files.

        :param descriptor_set: Index in which DescriptorElements will be
            stored.
        :param index_filepath: Optional file location to load/store MRPT index
            when initialized and/or built. If not configured, no model files
            are written to or loaded from disk.
        :param parameters_filepath: Optional file location to load/save index
            parameters determined at build time. If not configured, no model
            files are written to or loaded from disk.
        :param read_only: If True, `build_index` will error if there is an
            existing index. False by default.
        :param num_trees: The number of trees that will be generated for the
            data structure
        :param depth: The depth of the trees
        :param random_seed: Integer to use as the random number generator
            seed.
        :param pickle_protocol: The protocol version to be used by the pickle
            module to serialize class information
        :param use_multiprocessing: Whether or not to use discrete processes
            as the parallelization agent vs python threads.

        """
        super(MRPTNearestNeighborsIndex, self).__init__()

        self._read_only = read_only
        self._use_multiprocessing = use_multiprocessing
        self._descriptor_set = descriptor_set
        self._pickle_protocol = pickle_protocol

        def normpath(p: Optional[str]) -> Optional[str]:
            return (p and osp.abspath(osp.expanduser(p))) or p

        # Lock for model component access.
        self._model_lock = threading.RLock()

        self._index_filepath = normpath(index_filepath)
        self._index_param_filepath = normpath(parameters_filepath)
        # Now they're either None or an absolute path

        # parameters for building an index
        if depth < 1:
            raise ValueError("The depth may not be less than 1.")
        self._depth = depth
        if num_trees < 1:
            raise ValueError("The number of trees must be positive.")
        self._num_trees = num_trees

        # Set the list of trees to an empty list to have a sane value
        self._trees: List[TreeElement] = []

        self._rand_seed: Optional[int] = None
        if random_seed:
            self._rand_seed = int(random_seed)

        # Load the index/parameters if one exists
        if self._has_model_files():
            LOG.debug("Found existing model files. Loading.")
            self._load_mrpt_model()

    def get_config(self) -> Dict[str, Any]:
        return {
            "descriptor_set": to_config_dict(self._descriptor_set),
            "index_filepath": self._index_filepath,
            "parameters_filepath": self._index_param_filepath,
            "read_only": self._read_only,
            "random_seed": self._rand_seed,
            "pickle_protocol": self._pickle_protocol,
            "use_multiprocessing": self._use_multiprocessing,
            "depth": self._depth,
            "num_trees": self._num_trees,
        }

    def _has_model_files(self) -> bool:
        """
        check if configured model files are configured and exist
        """
        return bool(self._index_filepath and
                    osp.isfile(self._index_filepath) and
                    self._index_param_filepath and
                    osp.isfile(self._index_param_filepath))

    def _build_multiple_trees(self, chunk_size: int = CHUNK_SIZE) -> None:
        """
        Build an MRPT structure
        """
        sample = next(self._descriptor_set.iterdescriptors())
        sample_v = sample.vector()
        if sample_v is None:
            raise RuntimeError(
                "Sample descriptor element from the current set had no vector "
                "content!"
            )
        n = self.count()
        d = sample_v.size
        leaf_size = n / (1 << self._depth)

        nt = self._num_trees
        depth = self._depth
        LOG.debug(
            f"Building {nt} trees (T) of depth {depth} (l) "
            f"from {n:g} descriptors (N) of length {d:g}")
        LOG.debug(f"Leaf size             (L = N/2^l)  ~ {n:g}/2^{depth:d} = {leaf_size:g}")
        LOG.debug(f"UUIDs stored                (T*N)  = {nt:g} * {n:g} = {nt*n:g}")
        LOG.debug(f"Examined UUIDs              (T*L)  ~ {nt:g} * {leaf_size:g} = {nt*leaf_size:g}")
        LOG.debug(f"Examined/DB size  (T*L/N = T/2^l)  ~ {nt*leaf_size}/{n} = {nt*leaf_size/n:.3f}")

        if (1 << self._depth) > n:
            LOG.warning(
                f"There are insufficient elements ({n:d} < 2^{depth:d}) to "
                f"populate all the leaves of the tree. Consider lowering the "
                f"depth parameter.")

        LOG.debug("Projecting onto random bases")
        # Build all the random bases and the projections at the same time
        # (_num_trees * _depth shouldn't really be that high -- if it is,
        # you're a monster)
        rs = np.random.RandomState()
        if self._rand_seed is not None:
            rs.seed(self._rand_seed)
        random_bases = rs.randn(self._num_trees, d, self._depth)
        projs = np.empty((n, self._num_trees, self._depth), dtype=np.float64)
        # Load the data in chunks (because n * d IS high)
        pts_array = np.empty((chunk_size, d), sample_v.dtype)
        # Enumerate the descriptors and div the index by the chunk size
        # (causes each loop to only deal with at most chunk_size descriptors at
        # a time).
        for k, g in groupby(enumerate(self._descriptor_set.iterdescriptors()),
                            lambda pair: pair[0] // chunk_size):
            # Items are still paired so extract the descriptors
            chunk = list(desc for (i, desc) in g)
            # Take care of dangling end piece
            k_beg = k * chunk_size
            k_end = min((k+1) * chunk_size, n)
            k_len = k_end - k_beg
            # Run the descriptors through elements_to_matrix
            # - Using slicing on pts_array due to g being <= chunk-size on the
            #   last chunk.
            pts_array[:len(chunk)] = list(parallel_map(
                lambda d_: d_.vector(),
                chunk,
                use_multiprocessing=self._use_multiprocessing
            ))
            # Insert into projection matrix
            projs[k_beg:k_end] = pts_array[:k_len].dot(random_bases)
        del pts_array

        LOG.debug("Constructing trees")
        desc_ids = list(self._descriptor_set.keys())
        # Start with no trees
        self._trees = []
        for t in range(self._num_trees):
            # Array of splits is a packed tree
            splits = np.empty(((1 << self._depth) - 1,), np.float64)

            LOG.debug(f"Constructing tree #{t+1}")

            # Build the tree & store it
            leaves = self._build_single_tree(projs[:, t], splits)
            leaves_ids = [[desc_ids[idx] for idx in cast(Iterable[int], leaf)]
                          for leaf in leaves]
            self._trees.append(TreeElement(**{
                'random_basis': (random_bases[t]),
                'splits': splits,
                'leaves': leaves_ids,
            }))

    def _build_single_tree(self, proj: np.ndarray, splits: np.ndarray) -> List[np.ndarray]:
        """
        Build a single RP tree for fast kNN search

        :param proj: Projections of the dataset for this tree as an array of
            shape (N, levels).

        :param splits: (2^depth-1) array of splits corresponding to leaves
                       (tree, where immediate descendants follow parents;
                       index i's children are 2i+1 and 2i+2

        :return: Tree of splits and list of index arrays for each leaf
        """
        def _build_recursive(
            indices: np.ndarray,
            level: int = 0,
            split_index: int = 0
        ) -> List[np.ndarray]:
            """
            Descend recursively into tree to build it, setting splits and
            returning indices for leaves

            :param indices: The current array of (integer) indices before
                partitioning.
            :param level: The level in the tree
            :param split_index: The index of the split to set

            :return: A list of integer arrays representing leaf membership of
                source descriptors.
            """
            # If we're at the bottom, no split, just return the set
            if level == self._depth:
                return [indices]

            n = indices.size
            # If we literally don't have enough to populate the leaf, make it
            # empty
            if n < 1:
                return []

            # Get the random projections for these indices at this level
            # NB: Recall that the projection matrix has shape (levels, N)
            level_proj = proj[indices, level]

            # Split at the median if even, put median in upper half if not
            n_split = n // 2
            if n % 2 == 0:
                part_indices = np.argpartition(
                    level_proj, (n_split - 1, n_split))
                split_val = level_proj[part_indices[n_split - 1]]
                split_val += level_proj[part_indices[n_split]]
                split_val /= 2.0
            else:
                part_indices = np.argpartition(level_proj, n_split)
                split_val = level_proj[part_indices[n_split]]

            splits[split_index] = split_val

            # part_indices is relative to this block of values, recover
            # main indices
            left_indices = indices[part_indices[:n_split]]
            right_indices = indices[part_indices[n_split:]]

            # Descend into each split and get sub-splits
            left_out = _build_recursive(left_indices, level=level + 1,
                                        split_index=2 * split_index + 1)
            right_out = _build_recursive(right_indices, level=level + 1,
                                         split_index=2 * split_index + 2)

            # Assemble index set
            left_out.extend(right_out)
            return left_out

        return _build_recursive(np.arange(proj.shape[0]))

    def _save_mrpt_model(self) -> None:
        LOG.debug(f"Caching index and parameters: {self._index_filepath}, "
                  f"{self._index_param_filepath}")
        if self._index_filepath:
            LOG.debug(f"Caching index: {self._index_filepath}")
            safe_create_dir(osp.dirname(self._index_filepath))
            # noinspection PyTypeChecker
            with open(self._index_filepath, "wb") as f:
                pickle.dump(self._trees, f, self._pickle_protocol)
        if self._index_param_filepath:
            LOG.debug(f"Caching index params: {self._index_param_filepath}")
            safe_create_dir(osp.dirname(self._index_param_filepath))
            params = {
                "read_only": self._read_only,
                "num_trees": self._num_trees,
                "depth": self._depth,
            }
            # noinspection PyTypeChecker
            with open(self._index_param_filepath, "wb") as f:
                pickle.dump(params, f, self._pickle_protocol)

    def _load_mrpt_model(self) -> None:
        LOG.debug(f"Loading index and parameters: {self._index_filepath}, "
                  f"{self._index_param_filepath}")
        if self._index_param_filepath:
            LOG.debug(f"Loading index params: {self._index_param_filepath}")
            with open(self._index_param_filepath, 'rb') as f:
                params = pickle.load(f)
            self._read_only = params['read_only']
            self._num_trees = params['num_trees']
            self._depth = params['depth']

        # Load the index
        if self._index_filepath:
            LOG.debug(f"Loading index: {self._index_filepath}")
            # noinspection PyTypeChecker
            with open(self._index_filepath, "rb") as f:
                self._trees = pickle.load(f)

    def count(self) -> int:
        # Descriptor-set should already handle concurrency.
        return len(self._descriptor_set)

    def _build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        with self._model_lock:
            if self._read_only:
                raise ReadOnlyError("Cannot modify container attributes due to "
                                    "being in read-only mode.")

            LOG.info("Building new MRPT index")

            LOG.debug("Clearing and adding new descriptor elements")
            # NOTE: It may be the case for some DescriptorSet implementations,
            # this clear may interfere with iteration when part of the input
            # iterator of descriptors was this index's previous descriptor-set,
            # as is the case with ``update_index``.
            self._descriptor_set.clear()
            self._descriptor_set.add_many_descriptors(descriptors)

            LOG.debug('Building MRPT index')
            self._build_multiple_trees()

            self._save_mrpt_model()

    def _update_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        with self._model_lock:
            if self._read_only:
                raise ReadOnlyError("Cannot modify container attributes due "
                                    "to being in read-only mode.")
            LOG.debug("Updating index by rebuilding with union. ")
            self.build_index(chain(self._descriptor_set, descriptors))

    def _remove_from_index(self, uids: Iterable[Hashable]) -> None:
        with self._model_lock:
            if self._read_only:
                raise ReadOnlyError("Cannot modify container attributes due "
                                    "to being in read-only mode.")
            self._descriptor_set.remove_many_descriptors(uids)
            self.build_index(self._descriptor_set)

    def _nn(
        self,
        d: DescriptorElement,
        n: int = 1
    ) -> Tuple[Tuple[DescriptorElement, ...], Tuple[float, ...]]:
        # Parent template method already checks that `d` has a non-None vector
        d_v = d.vector()

        def _query_single(tree: TreeElement) -> List[Hashable]:
            # Search a single tree for the leaf that matches the query
            # NB: random_basis has shape (levels, N)
            random_basis = tree.random_basis
            assert d_v is not None
            proj_query = d_v.dot(random_basis)
            splits = tree.splits
            idx = 0
            for level in range(depth):
                split_point = splits[idx]
                # Look at the level'th coordinate of proj_query
                if proj_query[level] < split_point:
                    idx = 2 * idx + 1
                else:
                    idx = 2 * idx + 2

            # idx will be `2^depth - 1` greater than the position of the leaf
            # in the list
            idx -= ((1 << depth) - 1)
            return tree.leaves[idx]

        def _exact_query(_uuids: Sequence[Hashable]) -> Tuple[Sequence[Hashable], np.ndarray]:
            set_size = len(_uuids)
            LOG.debug(f"Exact query requested with {set_size} descriptors")

            # Assemble the array to query from the descriptors that match
            assert d_v is not None
            pts_array = np.empty((set_size, d_v.size), dtype=d_v.dtype)
            descriptors = self._descriptor_set.get_many_descriptors(_uuids)
            for i, desc in enumerate(descriptors):
                pts_array[i, :] = desc.vector()

            dists: np.ndarray = ((pts_array - d_v) ** 2).sum(axis=1)

            if n > dists.shape[0]:
                LOG.warning(
                    f"There were fewer descriptors ({dists.shape[0]}) in the "
                    f"set than requested in the query ({n}). Returning entire "
                    f"set.")
            if n >= dists.shape[0]:
                return _uuids, dists

            near_indices = np.argpartition(dists, n - 1)[:n]
            return ([_uuids[idx] for idx in near_indices],
                    dists[near_indices])

        with self._model_lock:
            LOG.debug(f"Received query for {n} nearest neighbors")

            depth, ntrees, db_size = self._depth, self._num_trees, self.count()
            leaf_size = db_size//(1 << depth)
            if leaf_size * ntrees < n:
                LOG.warning(
                    f"The number of descriptors in a leaf ({leaf_size}) times "
                    f"the number of trees ({ntrees}) is less than the number "
                    f"of descriptors requested by the query ({n}). The query "
                    f"result will be deficient.")

            # Take union of all tree hits
            tree_hits: Set[Hashable] = set()
            for t in self._trees:
                tree_hits.update(_query_single(t))

            hit_union = len(tree_hits)
            LOG.debug(
                f"Query (k): {n}, Hit union (h): {hit_union}, "
                f"DB (N): {db_size}, Leaf size (L = N/2^l): {leaf_size}, "
                f"Examined (T*L): {leaf_size * ntrees}")
            LOG.debug(f"k/L     = {n / leaf_size:.3f}")
            LOG.debug(f"h/N     = {hit_union / db_size:.3f}")
            LOG.debug(f"h/L     = {hit_union / leaf_size:.3f}")
            LOG.debug(f"h/(T*L) = {hit_union / (leaf_size * ntrees):.3f}")

            uuids, distances = _exact_query(list(tree_hits))
            order = distances.argsort()
            uuids, distances = zip(
                *((uuids[oidx], distances[oidx]) for oidx in order))

            LOG.debug(f"Returning query result of size {len(uuids)}")

            return (tuple(self._descriptor_set.get_many_descriptors(uuids)),
                    tuple(distances))

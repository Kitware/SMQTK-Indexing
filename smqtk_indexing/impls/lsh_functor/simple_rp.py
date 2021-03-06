import logging
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from smqtk_descriptors import DescriptorElement
from smqtk_descriptors.utils import parallel_map
from smqtk_indexing import LshFunctor
from smqtk_indexing.utils.progress_reporter import ProgressReporter


LOG = logging.getLogger(__name__)


class SimpleRPFunctor (LshFunctor):
    """
    This class is meant purely as a baseline comparison for other
    LshFunctors and NNIndex plugins. It is not meant to be used in
    production, as it is unlikely to produce a quality index.
    """

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def __init__(
        self,
        bit_length: int = 8,
        normalize: Optional[Union[int, float, str]] = None,
        random_seed: Optional[int] = None
    ):
        super(SimpleRPFunctor, self).__init__()

        self.bit_length = bit_length
        self.normalize = normalize
        self.random_seed = random_seed

        # Model components
        self.rps = None
        self.mean_vec = None

    def _norm_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Class standard array normalization. Normalized along max dimension
        (a=0 for a 1D array, a=1 for a 2D array, etc.).

        :param v: Vector to normalize

        :return: Returns the normalized version of input array ``v``.

        """
        vm = v - self.mean_vec
        if self.normalize is None:
            # Normalization off
            return vm

        n = np.linalg.norm(vm, ord=self.normalize, axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            return np.nan_to_num(vm/n)

    def get_config(self) -> Dict[str, Any]:
        return {
            "bit_length": self.bit_length,
            "normalize": self.normalize,
            "random_seed": self.random_seed,
        }

    def has_model(self) -> bool:
        return self.mean_vec is not None

    def fit(
        self,
        descriptors: Iterable[DescriptorElement],
        use_multiprocessing: bool = True
    ) -> np.ndarray:
        """
        Fit the ITQ model given the input set of descriptors

        :param descriptors: Iterable of ``DescriptorElement`` vectors to fit
            the model to.
        :param use_multiprocessing: If multiprocessing should be used, as
            opposed to threading, for collecting descriptor vectors from the
            provided iterable.

        :raises RuntimeError: There is already a model loaded

        :return: Matrix hash code vectors (boolean-typed) for provided
            descriptors in order.

        """
        if self.has_model():
            raise RuntimeError("Model components have already been loaded.")

        dbg_report_interval = None
        pr = None
        if LOG.getEffectiveLevel() <= logging.DEBUG:
            dbg_report_interval = 1.0  # seconds
            pr = ProgressReporter(LOG.debug, dbg_report_interval).start()
        if not hasattr(descriptors, "__len__"):
            LOG.info("Creating sequence from iterable")
            descriptors_l = []
            for d in descriptors:
                descriptors_l.append(d)
                dbg_report_interval and pr.increment_report()  # type: ignore
            dbg_report_interval and pr.report()  # type: ignore
            descriptors = descriptors_l
        LOG.info("Creating matrix of descriptors for fitting")
        x = np.asarray(list(
            parallel_map(lambda d_: d_.vector(), descriptors,
                         use_multiprocessing=use_multiprocessing)
        ))
        LOG.debug("descriptor matrix shape: %s", x.shape)
        n, dim = x.shape

        LOG.debug("Generating random projections")
        np.random.seed(self.random_seed)
        self.rps = np.random.randn(dim, self.bit_length)

        LOG.debug(f"Info normalizing descriptors with norm type: {self.normalize}")
        return self.get_hash(x)

    def get_hash(self, descriptor: np.ndarray) -> np.ndarray:
        if self.rps is None:
            raise RuntimeError("Random projection model not constructed. Call "
                               "`fit` first!")
        b = (self._norm_vector(descriptor).dot(self.rps) >= 0.0)
        return b.squeeze()

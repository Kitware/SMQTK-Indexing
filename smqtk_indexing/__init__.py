import pkg_resources

from .interfaces.nearest_neighbor_index import NearestNeighborsIndex  # noqa: F401
from .interfaces.lsh_functor import LshFunctor  # noqa: F401
from .interfaces.hash_index import HashIndex  # noqa: F401


# It is known that this will fail if this package is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version

Algorithm Interfaces
--------------------
Here we list and briefly describe the high level algorithm interfaces which this
SMQTK-Indexing package provides.
There is at least one implementation available for each interface.
Some implementations will require additional dependencies.


NearestNeighborsIndex
+++++++++++++++++++++
This interface defines a method to build an index from a set of
:class:`smqtk_descriptors.DescriptorElement` instances (:meth:`.NearestNeighborsIndex.build_index`)
and a nearest-neighbors query function for getting a number of near neighbors
to a query :class:`.DescriptorElement` (:meth:`.NearestNeighborsIndex.nn`).

Building an index requires that some non-zero number of
:class:`.DescriptorElement` instances be passed into the
:meth:`.NearestNeighborsIndex.build_index` method.
Subsequent calls to this method should rebuild the index model, not add to it.
If an implementation supports persistent storage of the index, it should
overwrite the configured index.

The :meth:`.NearestNeighborsIndex.nn` method uses a single
:class:`.DescriptorElement` to query the current index for a specified number
of nearest neighbors.
Thus, the :class:`.NearestNeighborsIndex` instance must have a non-empty index
loaded for this method to function.
If the provided query :class:`.DescriptorElement` does not have a set vector,
this method will also fail with an exception.

This interface additionally requires that implementations define a
:meth:`.NearestNeighborsIndex.count` method, which returns the number of distinct
:class:`.DescriptorElement` instances are in the index.

.. autoclass:: smqtk_indexing.interfaces.nearest_neighbor_index.NearestNeighborsIndex
   :members:


LshFunctor
++++++++++
Implementations of this interface define the generation of a locality-sensitive
hash code for a given :class:`.DescriptorElement`.
These are used in :class:`smqtk_indexing.impls.nn_index.lsh.LSHNearestNeighborIndex`
instances.

.. autoclass:: smqtk_indexing.interfaces.lsh_functor.LshFunctor
   :members:


HashIndex
+++++++++
This interface describes specialized :class:`.NearestNeighborsIndex`
implementations designed to index hash codes (bit vectors) via the hamming
distance metric function.
Implementations of this interface are primarily used with the
:class:`smqtk_indexing.impls.nn_index.lsh.LSHNearestNeighborIndex`
implementation.

Unlike the :class:`.NearestNeighborsIndex` interface from which this interface
is very similar to, :class:`.HashIndex` instances are built with an iterable of
:class:`numpy.ndarray` and :meth:`.HashIndex.nn` returns a
:class:`numpy.ndarray`.

.. autoclass:: smqtk_indexing.interfaces.hash_index.HashIndex
   :members:

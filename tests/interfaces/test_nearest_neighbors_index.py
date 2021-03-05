from typing import Any, Dict, Hashable, Iterable, Tuple
import unittest
import unittest.mock as mock

import numpy

from smqtk_descriptors import DescriptorElement
from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement
from smqtk_indexing.interfaces.nearest_neighbor_index import NearestNeighborsIndex
from smqtk_indexing.utils.iter_validation import check_empty_iterable


class DummySI (NearestNeighborsIndex):

    @classmethod
    def is_usable(cls) -> bool:
        """ stub """
        return True

    def get_config(self) -> Dict[str, Any]:
        """ stub """

    def _build_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """ stub """

    def _update_index(self, descriptors: Iterable[DescriptorElement]) -> None:
        """ stub """

    def _remove_from_index(self, uids: Iterable[Hashable]) -> None:
        """ stub """

    def _nn(
        self,
        d: DescriptorElement,
        n: int = 1
    ) -> Tuple[Tuple[DescriptorElement, ...], Tuple[float, ...]]:
        """ stub """

    def count(self) -> int:
        return 0


class TestNNIndexAbstract (unittest.TestCase):

    def test_get_impls(self) -> None:
        # Some implementations should be returned
        m = NearestNeighborsIndex.get_impls()
        self.assertTrue(m)
        for cls in m:
            self.assertTrue(issubclass(cls, NearestNeighborsIndex))

    def test_empty_iterable_exception(self) -> None:
        v = DummySI._empty_iterable_exception()
        self.assertIsInstance(v, ValueError)
        self.assertRegex(str(v), "DescriptorElement")

    def test_check_empty_iterable_no_data(self) -> None:
        # Test that an exception is thrown when an empty list/iterable is
        # passed.  Additionally check that the exception thrown has expected
        # message from exception generation method.
        callback = mock.MagicMock()

        # Not-stateful iterable (list)
        self.assertRaisesRegex(
            ValueError,
            str(DummySI._empty_iterable_exception()),
            check_empty_iterable, [], callback,
            DummySI._empty_iterable_exception()
        )
        callback.assert_not_called()

        # with a stateful iterator.
        self.assertRaisesRegex(
            ValueError,
            str(DummySI._empty_iterable_exception()),
            check_empty_iterable, iter([]), callback,
            DummySI._empty_iterable_exception()
        )
        callback.assert_not_called()

    def test_check_empty_iterable_valid_iterable(self) -> None:
        # Test that the method correctly calls the callback with the full
        # iterable when what is passed is not empty.
        callback = mock.MagicMock()

        # non-stateful iterator (set)
        d_set = {0, 1, 2, 3, 4}
        check_empty_iterable(d_set, callback,
                             DummySI()._empty_iterable_exception())
        callback.assert_called_once()
        self.assertSetEqual(
            set(callback.call_args[0][0]),
            d_set
        )

        # Stateful iterator
        callback = mock.MagicMock()
        check_empty_iterable(iter(d_set), callback,
                             DummySI()._empty_iterable_exception())
        callback.assert_called_once()
        self.assertSetEqual(
            set(callback.call_args[0][0]),
            d_set
        )

    def test_count_and_len(self) -> None:
        index = DummySI()
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.count(), len(index))

        # Pretend that there were things in there. Len should pass it though
        # noinspection PyTypeHints
        index.count = mock.Mock()  # type: ignore
        index.count.return_value = 5
        self.assertEqual(len(index), 5)

    def test_build_index_no_descriptors(self) -> None:
        index = DummySI()
        # noinspection PyTypeHints
        index._build_index = mock.MagicMock()  # type: ignore
        self.assertRaises(
            ValueError,
            index.build_index,
            []
        )
        index._build_index.assert_not_called()

    def test_build_index_nonzero_descriptors(self) -> None:
        index = DummySI()
        # noinspection PyTypeHints
        index._build_index = mock.MagicMock()  # type: ignore
        d = DescriptorMemoryElement('test', 0)
        index.build_index([d])
        index._build_index.assert_called_once()
        # Check that the last call's first (only) argument was the same iterable
        # given.
        self.assertSetEqual(
            set(index._build_index.call_args[0][0]),
            {d}
        )

    def test_build_index_iterable(self) -> None:
        # Test build check with a pure iterable
        index = DummySI()
        # noinspection PyTypeHints
        index._build_index = mock.MagicMock()  # type: ignore
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        it = iter(d_set)
        index.build_index(it)
        # _build_index should have been called and the contents of the iterable
        # it was called with should equal d_set.
        index._build_index.assert_called_once()
        self.assertSetEqual(
            set(index._build_index.call_args[0][0]),
            d_set
        )

    def test_update_index_no_descriptors(self) -> None:
        index = DummySI()
        # noinspection PyTypeHints
        index._update_index = mock.MagicMock()  # type: ignore
        self.assertRaises(
            ValueError,
            index.update_index,
            []
        )
        # internal method should not have been called.
        index._update_index.assert_not_called()

    def test_update_index_nonzero_descriptors(self) -> None:
        index = DummySI()
        # noinspection PyTypeHints
        index._update_index = mock.MagicMock()  # type: ignore

        # Testing with dummy input data.
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        index.update_index(d_set)
        index._update_index.assert_called_once()
        self.assertSetEqual(
            set(index._update_index.call_args[0][0]),
            d_set
        )

    def test_update_index_iterable(self) -> None:
        # Test build check with a pure iterable.
        index = DummySI()
        # noinspection PyTypeHints
        index._update_index = mock.MagicMock()  # type: ignore
        d_set = {
            DescriptorMemoryElement('test', 0),
            DescriptorMemoryElement('test', 1),
            DescriptorMemoryElement('test', 2),
            DescriptorMemoryElement('test', 3),
        }
        it = iter(d_set)
        index.update_index(it)

        index._update_index.assert_called_once()
        self.assertSetEqual(
            set(index._update_index.call_args[0][0]),
            d_set,
        )

    def test_remove_from_index_no_uids(self) -> None:
        # Test that the method errors when no UIDs are provided
        index = DummySI()
        # noinspection PyTypeHints
        index._remove_from_index = mock.Mock()  # type: ignore
        self.assertRaises(
            ValueError,
            index.remove_from_index, []
        )
        index._remove_from_index.assert_not_called()

    def test_remove_from_index_nonzero_descriptors(self) -> None:
        # Test removing a non-zero amount of descriptors
        index = DummySI()
        # noinspection PyTypeHints
        index._remove_from_index = mock.MagicMock()  # type: ignore

        # Testing with dummy input data.
        uid_set = {0, 1, 2, 3}
        index.remove_from_index(uid_set)
        index._remove_from_index.assert_called_once()
        self.assertSetEqual(
            set(index._remove_from_index.call_args[0][0]),
            uid_set
        )

    def test_remove_from_index_nonzero_iterable(self) -> None:
        # Test removing a non-zero amount of descriptors via an iterable.
        index = DummySI()
        # noinspection PyTypeHints
        index._remove_from_index = mock.MagicMock()  # type: ignore
        d_set = {0, 1, 2, 3}
        it = iter(d_set)
        index.remove_from_index(it)

        index._remove_from_index.assert_called_once()
        self.assertSetEqual(
            set(index._remove_from_index.call_args[0][0]),
            d_set,
        )

    def test_nn_empty_vector(self) -> None:
        # ValueError should be thrown if the input element has no vector.
        index = DummySI()
        # Need to force a non-zero index size for knn to be performed.
        # noinspection PyTypeHints
        index.count = mock.MagicMock(return_value=1)  # type: ignore
        # Observe internal function
        # noinspection PyTypeHints
        index._nn = mock.MagicMock()  # type: ignore

        q = DescriptorMemoryElement('test', 0)
        self.assertRaises(
            ValueError,
            index.nn, q
        )
        # template method should not have been called.
        index._nn.assert_not_called()

    def test_nn_empty_index(self) -> None:
        # nn should fail if index size is 0
        index = DummySI()
        # noinspection PyTypeHints
        index.count = mock.MagicMock(return_value=0)  # type: ignore
        # noinspection PyTypeHints
        index._nn = mock.MagicMock()  # type: ignore

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        self.assertRaises(
            ValueError,
            index.nn, q
        )

    # noinspection PyUnresolvedReferences
    def test_nn_normal_conditions(self) -> None:
        index = DummySI()
        # Need to force a non-zero index size for knn to be performed.
        # noinspection PyTypeHints
        index.count = mock.MagicMock()  # type: ignore
        index.count.return_value = 1

        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        # Basically this shouldn't crash
        index.nn(q)

    def test_query_empty_index(self) -> None:
        index = DummySI()
        q = DescriptorMemoryElement('q', 0)
        q.set_vector(numpy.random.rand(4))
        self.assertRaises(ValueError, index.nn, q)

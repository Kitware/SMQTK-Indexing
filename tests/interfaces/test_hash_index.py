from typing import Any, Dict, Iterable, Tuple
import unittest.mock as mock
import unittest

import numpy as np

from smqtk_indexing.interfaces.hash_index import HashIndex


class DummyHI (HashIndex):

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:  # type: ignore[empty-body]
        """ stub """

    def count(self) -> int:
        return 0

    def _build_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _update_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _remove_from_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _nn(self, h: np.ndarray, n: int = 1) -> Tuple[np.ndarray, Tuple[float, ...]]:  # type: ignore[empty-body]
        """ stub """


class TestHashIndex (unittest.TestCase):

    def test_empty_iterable_exception(self) -> None:
        v = DummyHI._empty_iterable_exception()
        self.assertIsInstance(v, ValueError)
        self.assertRegex(str(v), "hash vectors")

    def test_build_index_empty_iter(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._build_index = mock.MagicMock()  # type: ignore
        self.assertRaisesRegex(
            ValueError,
            str(HashIndex._empty_iterable_exception()),
            idx.build_index, []
        )
        # Internal method should not have been called
        idx._build_index.assert_not_called()

    def test_build_index_with_values(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._build_index = mock.MagicMock()  # type: ignore
        # No error should be returned. Returned iterable contents should match
        # input values.
        a = np.array([0])
        b = np.array([1])
        c = np.array([2])
        idx.build_index([a, b, c])
        self.assertListEqual(
            list(idx._build_index.call_args[0][0]),
            [a, b, c]
        )

    def test_update_index_empty_iter(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._update_index = mock.MagicMock()  # type: ignore
        self.assertRaisesRegex(
            ValueError,
            "No hash vectors.*",
            idx.update_index, []
        )
        # Internal method should not have been called.
        idx._update_index.assert_not_called()

    def test_update_index_with_values(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._update_index = mock.MagicMock()  # type: ignore
        # No error should be returned. Returned iterable contents should match
        # input values.
        a = np.array([0])
        b = np.array([1])
        c = np.array([2])
        idx.update_index([a, b, c])
        self.assertListEqual(
            list(idx._update_index.call_args[0][0]),
            [a, b, c]
        )

    def test_remove_from_index_empty_iter(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._remove_from_index = mock.MagicMock()  # type: ignore
        self.assertRaisesRegex(
            ValueError,
            "No hash vectors.*",
            idx.update_index, []
        )
        # Internal method should not have been called.
        idx._remove_from_index.assert_not_called()

    def test_remove_from_index_with_values(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._remove_from_index = mock.MagicMock()  # type: ignore
        # No error should be returned. Returned iterable contents should match
        # input values.
        a = np.array([0])
        b = np.array([1])
        c = np.array([2])
        idx._remove_from_index([a, b, c])
        self.assertListEqual(
            list(idx._remove_from_index.call_args[0][0]),
            [0, 1, 2]
        )

    def test_nn_no_index(self) -> None:
        idx = DummyHI()
        # noinspection PyTypeHints
        idx._nn = mock.MagicMock()  # type: ignore
        self.assertRaises(
            ValueError,
            idx.nn, 'something'
        )
        # Internal method should not have been called.
        idx._nn.assert_not_called()

    def test_nn_has_count(self) -> None:
        idx = DummyHI()
        idx.count = mock.MagicMock()  # type: ignore
        idx.count.return_value = 10
        idx._nn = mock.MagicMock()  # type: ignore
        # This call should now pass that count returns something greater than 0.
        test_query = np.array([1, 2, 3])
        idx.nn(test_query)
        idx._nn.assert_called_with(test_query, 1)

        idx.nn(test_query, 10)
        idx._nn.assert_called_with(test_query, 10)

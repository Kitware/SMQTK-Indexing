from typing import Any, Dict, Iterable, Tuple
import unittest.mock as mock
import unittest

import numpy as np

from smqtk_indexing.interfaces.hash_index import HashIndex


class DummyHI (HashIndex):

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:
        """ stub """

    def count(self) -> int:
        return 0

    def _build_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _update_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _remove_from_index(self, hashes: Iterable[np.ndarray]) -> None:
        """ stub """

    def _nn(self, h: np.ndarray, n: int = 1) -> Tuple[np.ndarray, Tuple[float, ...]]:
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
        # noinspection PyTypeChecker
        idx.build_index([0, 1, 2])
        self.assertSetEqual(
            set(idx._build_index.call_args[0][0]),
            {0, 1, 2}
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
        # noinspection PyTypeChecker
        idx.update_index([0, 1, 2])
        self.assertSetEqual(
            set(idx._update_index.call_args[0][0]),
            {0, 1, 2}
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
        # noinspection PyTypeChecker
        idx._remove_from_index([0, 1, 2])
        self.assertSetEqual(
            set(idx._remove_from_index.call_args[0][0]),
            {0, 1, 2}
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
        # noinspection PyTypeHints
        idx.count = mock.MagicMock()  # type: ignore
        idx.count.return_value = 10
        # noinspection PyTypeHints
        idx._nn = mock.MagicMock()  # type: ignore
        # This call should now pass that count returns something greater than 0.
        # noinspection PyTypeChecker
        idx.nn('dummy')
        idx._nn.assert_called_with("dummy", 1)

        # noinspection PyTypeChecker
        idx.nn('bar', 10)
        idx._nn.assert_called_with("bar", 10)

from typing import Any, Dict
import unittest
import unittest.mock as mock

import numpy as np

from smqtk_indexing.interfaces.lsh_functor import LshFunctor


class DummyLshFunctor (LshFunctor):

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:
        pass

    def get_hash(self, descriptor: np.ndarray) -> np.ndarray:
        pass


class TestLshFunctorAbstract (unittest.TestCase):

    def test_call(self) -> None:
        # calling an instance should get us to the get_hash method.
        f = DummyLshFunctor()
        # noinspection PyTypeHints
        f.get_hash = mock.MagicMock()  # type: ignore

        expected_descriptor = 'pretend descriptor element'
        # noinspection PyTypeChecker
        f(expected_descriptor)
        f.get_hash.assert_called_once_with(expected_descriptor)

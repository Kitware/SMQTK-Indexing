import unittest

import numpy

from smqtk_indexing.utils import bits


class TestBitUtils (unittest.TestCase):

    def test_int_to_bit_vector_large_0(self):
        # Need at least one bit to represent 0.
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(0),
            [False]
        )
        # Force 5 bits.
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(0, 5),
            [False, False, False, False, False]
        )

    def test_int_to_bit_vector_large_1(self):
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(1),
            [True]
        )
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(1, 7),
            ([False] * 6) + [True]
        )

    def test_int_to_bit_vector_large_large(self):
        # Try large integer bit vectors
        int_val = (2**256) - 1
        expected_vector = [True] * 256
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(int_val),
            expected_vector
        )

        int_val = (2**512)
        expected_vector = [True] + ([False] * 512)
        numpy.testing.assert_array_equal(
            bits.int_to_bit_vector_large(int_val),
            expected_vector
        )

    def test_int_to_bit_vector_large_invalid_bits(self):
        # Cannot represent 5 in binary using 1 bit.
        self.assertRaises(
            ValueError,
            bits.int_to_bit_vector_large,
            5, 1
        )

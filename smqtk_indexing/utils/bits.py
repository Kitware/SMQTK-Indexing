import numpy


def bit_vector_to_int_large(v: numpy.ndarray) -> int:
    """
    Transform a numpy vector representing a sequence of binary bits [0 | >0]
    into an integer representation.

    This function is the special form that can handle very large integers
    (>64bit).

    :param v: 1D Vector of bits

    :return: Integer equivalent

    """
    c = 0
    for b in v:
        c = (c << 1) + int(b)
    return c


def int_to_bit_vector_large(integer: int, bits: int = 0) -> numpy.ndarray:
    """
    Transform integer into a bit vector, optionally of a specific length.

    This function is the special form that can handle very large integers
    (>64bit).

    :raises ValueError: If ``bits`` specified is smaller than the required bits
        to represent the given ``integer`` value.

    :param integer: integer to convert
    :param bits: Optional fixed number of bits that should be represented by the
        vector.

    :return: Bit vector as numpy array (big endian).

    """
    # Can't use math version because floating-point precision runs out after
    # about 2^48
    # -2 to remove length of '0b' string prefix
    size = len(bin(integer)) - 2

    if bits and (bits - size) < 0:
        raise ValueError(
            "%d bits too small to represent integer value %d." % (bits, integer)
        )

    # Converting integer to array
    v = numpy.zeros(bits or size, numpy.bool_)
    for i in range(0, size):
        v[-(i + 1)] = integer & 1
        integer >>= 1

    return v

import itertools
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def check_empty_iterable(
    iterable: Iterable,
    callback: Callable[[Iterable], T],
    exception_inst: BaseException
) -> T:
    """
    Check that the given iterable is not empty, then call the given callback
    function with the reconstructed iterable when it is not empty.

    :param iterable: Iterable to check.
    :param callback: Function to call with the reconstructed, not-empty
        iterable.
    :param exception_inst: The exception to throw if the iterable is empty

    """
    i = iter(iterable)
    try:
        first = next(i)
    except StopIteration:
        raise exception_inst
    return callback(itertools.chain([first], i))

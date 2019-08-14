"""
Utilities for implementing 2D compatibility for 1D ExtensionArrays.
"""
from functools import wraps
from typing import Tuple
import warnings

import numpy as np

from pandas._libs.lib import is_integer

msg = (
    "ExtensionArray subclass {name} defines {method}. "
    "In the future, this will not be supported. Please "
    "inherit {method} from the ExtensionArray base class."
)


def implement_2d(cls):
    """
    Patch a 1-dimension-only ExtensionArray subclass and make
    it support limited 2-dimensional operations.

    We achieve this by rewriting dimension-dependent methods to
    pre-process the inputs to make them look 1d, call the underlying
    method, and post-process the output.
    """
    if cls._allows_2d:
        return

    if cls.__name__ == "ExtensionArray" and cls.__module__ == "pandas.core.arrays.base":
        # No need to patch for ExtensionArray base class.
        return
    else:
        from pandas.core.arrays import ExtensionArray

    # For backwards-compatibility, we use the length, size, or shape
    # defined by the subclass. We can always define the other two in
    # terms of the one.
    has_size = cls.size is not ExtensionArray.size
    has_shape = cls.shape is not ExtensionArray.shape

    orig_len = cls.__len__
    # TODO: Find a better way to do this. I suspect we could check whether
    # our cls.bases contains ExtensionArray...
    if hasattr(orig_len, "_original_len"):
        # When a user does class Foo(Bar(ExtensionArray)):
        # we want to use the unpatched verison.
        orig_len = orig_len._original_len

    orig_shape = cls.shape

    @wraps(orig_len)
    def __len__(self):
        length = orig_len(self)
        if self._ExtensionArray__expanded_dim is None:
            result = length
        elif self._ExtensionArray__expanded_dim == 0:
            result = length
        else:
            result = 1

        return result

    cls.__len__ = __len__
    cls.__len__._original_len = orig_len

    if has_shape:
        warnings.warn(msg.format(name=cls.__name__, method="shape"), DeprecationWarning)

        def get_shape(self):
            return ExtensionArray.shape.fget(self)

        def set_shape(self, value):
            if orig_shape.fset:
                orig_shape.fset(self, value)

            ExtensionArray.shape.fset(self, value)

        cls.shape = property(fget=get_shape, fset=set_shape)

    if has_size:
        warnings.warn(msg.format(name=cls.__name__, method="size"), DeprecationWarning)

        def get_size(self):
            return ExtensionArray.size.fget(self)

        cls.size = property(fget=get_size)

    orig_copy = cls.copy

    @wraps(orig_copy)
    def copy(self):
        result = orig_copy(self)
        # TODO: Can this setattr be done in the metaclass? Less likely to forget.
        result._ExtensionArray__expanded_dim = self._ExtensionArray__expanded_dim
        return result

    cls.copy = copy

    orig_getitem = cls.__getitem__

    def __getitem__(self, key):
        if self.ndim == 1:
            return orig_getitem(self, key)

        key = expand_key(key, self.shape)
        if is_integer(key[0]):
            assert key[0] in [0, -1]
            result = orig_getitem(self, key[1])
            return result

        if isinstance(key[0], slice):
            if slice_contains_zero(key[0]):
                result = orig_getitem(self, key[1])
                result._ExtensionArray__expanded_dim = 1
                return result

            raise NotImplementedError(key)
        # TODO: ellipses?
        raise NotImplementedError(key)

    cls.__getitem__ = __getitem__

    orig_take = cls.take

    # kwargs for compat with Interval
    # allow_fill=None instead of False is for compat with Categorical
    def take(self, indices, allow_fill=None, fill_value=None, axis=0, **kwargs):
        if self.ndim == 1 and axis == 0:
            return orig_take(
                self, indices, allow_fill=allow_fill, fill_value=fill_value, **kwargs
            )

        if self.ndim != 2 or self.shape[0] != 1:
            raise NotImplementedError
        if axis not in [0, 1]:
            raise ValueError(axis)
        if kwargs:
            raise ValueError(
                "kwargs should not be passed in the 2D case, "
                "are only included for compat with Interval"
            )

        if axis == 1:
            result = orig_take(
                self, indices, allow_fill=allow_fill, fill_value=fill_value
            )
            result._ExtensionArray__expanded_dim = 1
            return result

        # For axis == 0, because we only support shape (1, N)
        #  there are only limited indices we can accept
        if len(indices) != 1:
            # TODO: we could probably support zero-len here
            raise NotImplementedError

        def take_item(n):
            if n == -1:
                seq = [fill_value] * self.shape[1]
                return type(self)._from_sequence(seq)
            else:
                return self[n, :]

        arrs = [take_item(n) for n in indices]
        result = type(self)._concat_same_type(arrs)
        result.shape = (len(indices), self.shape[1])
        return result

    cls.take = take

    orig_iter = cls.__iter__

    def __iter__(self):
        if self.ndim == 1:
            for obj in orig_iter(self):
                yield obj
        else:
            for n in range(self.shape[0]):
                yield self[n]

    cls.__iter__ = __iter__

    return cls


def slice_contains_zero(slc: slice) -> bool:
    if slc == slice(None):
        return True
    if slc == slice(0, None):
        return True
    if slc == slice(0, 1):
        return True
    if slc.start == slc.stop:
        # Note: order matters here, since we _dont_ want this to catch
        #  the slice(None) case.
        return False
    raise NotImplementedError(slc)


def expand_key(key, shape):
    ndim = len(shape)
    if ndim != 2 or shape[0] != 1:
        raise NotImplementedError
    if not isinstance(key, tuple):
        key = (key, slice(None))
    if len(key) != 2:
        raise ValueError(key)

    if is_integer(key[0]) and key[0] not in [0, -1]:
        raise ValueError(key)

    return key


def can_safe_ravel(shape: Tuple[int, ...]) -> bool:
    """
    Check if an array with the given shape can be ravelled unambiguously
    regardless of column/row order.

    Parameters
    ----------
    shape : tuple[int]

    Returns
    -------
    bool
    """
    if len(shape) == 1:
        return True
    if len(shape) > 2:
        raise NotImplementedError(shape)
    if shape[0] == 1 or shape[1] == 1:
        # column-like or row-like
        return True
    return False


def tuplify_shape(size: int, shape, restrict=True) -> Tuple[int, ...]:
    """
    Convert a passed shape into a valid tuple.
    Following ndarray.reshape, we accept either `reshape(a, b)` or
    `reshape((a, b))`, the latter being canonical.

    Parameters
    ----------
    size : int
    shape : tuple
    restrict : bool, default True
        Whether to restrict to shapes (N), (1,N), and (N,1)

    Returns
    -------
    tuple[int, ...]
    """
    if len(shape) == 0:
        raise ValueError("shape must be a non-empty tuple of integers", shape)

    if len(shape) == 1:
        if is_integer(shape[0]):
            pass
        else:
            shape = shape[0]
            if not isinstance(shape, tuple):
                raise ValueError("shape must be a non-empty tuple of integers", shape)

    if not all(is_integer(x) for x in shape):
        raise ValueError("shape must be a non-empty tuple of integers", shape)

    if any(x < -1 for x in shape):
        raise ValueError("Invalid shape {shape}".format(shape=shape))

    if -1 in shape:
        if shape.count(-1) != 1:
            raise ValueError("Invalid shape {shape}".format(shape=shape))
        idx = shape.index(-1)
        others = [n for n in shape if n != -1]
        prod = np.prod(others)
        dim = size // prod
        shape = shape[:idx] + (dim,) + shape[idx + 1 :]

    if np.prod(shape) != size:
        raise ValueError(
            "Product of shape ({shape}) must match "
            "size ({size})".format(shape=shape, size=size)
        )

    num_gt1 = len([x for x in shape if x > 1])
    if num_gt1 > 1 and restrict:
        raise ValueError(
            "The default `reshape` implementation is limited to "
            "shapes (N,), (N,1), and (1,N), not {shape}".format(shape=shape)
        )
    return shape

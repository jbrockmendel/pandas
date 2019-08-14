import warnings

import numpy as np
import pytest

import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "ExtensionArray subclass", DeprecationWarning)

    class MyArray(pd.arrays.PandasArray):
        _typ = "extension"
        _allows_2d = False

        def __init__(self, values, copy=False):
            super().__init__(values, copy)
            self._length = len(self._ndarray)

    class LengthArray(MyArray):
        def __len__(self):
            return self._length

    class ShapeArray(MyArray):
        @property
        def shape(self):
            return (len(self),)

    class SizeArray(MyArray):
        @property
        def size(self):
            # This used to be correct. May not be correct now.
            return self.shape[0]

    class LengthShapeArray(LengthArray, ShapeArray):
        pass

    class LengthSizeArray(LengthArray, SizeArray):
        pass

    class ShapeSizeArray(ShapeArray, SizeArray):
        pass

    class LengthShapeSizeArray(LengthArray, ShapeArray, SizeArray):
        pass


@pytest.mark.parametrize(
    "cls",
    [
        MyArray,
        LengthArray,
        ShapeArray,
        SizeArray,
        LengthShapeArray,
        LengthSizeArray,
        ShapeSizeArray,
        LengthShapeSizeArray,
    ],
)
def test_attributes(cls):
    a = np.arange(10)
    array = cls(a)

    assert len(array) == 10
    assert array.shape == (10,)
    assert array.size == 10

    array2 = array.reshape(-1, 1)

    assert len(array2) == 10
    assert array2.shape == (10, 1)
    assert array2.size == 10

    array3 = array.reshape(1, -1)
    # assert len(array3) == 1
    assert array3.shape == (1, 10)
    assert array3.size == 10

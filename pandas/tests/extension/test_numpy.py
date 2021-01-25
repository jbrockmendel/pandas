"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import ExtensionDtype, PandasDtype
from pandas.core.dtypes.missing import infer_fill_value as infer_fill_value_orig

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PandasArray, StringArray
from pandas.core.construction import extract_array

from . import base


@pytest.fixture(params=["float", "object"])
def dtype(request):
    return PandasDtype(np.dtype(request.param))


orig_setitem = pd.core.internals.Block.setitem


def setitem(self, indexer, value):
    # patch Block.setitem
    value = extract_array(value, extract_numpy=True)
    if isinstance(value, PandasArray) and not isinstance(value, StringArray):
        value = value.to_numpy()
        if self.ndim == 2 and value.ndim == 1:
            # TODO(EA2D): special case not needed with 2D EAs
            value = np.atleast_2d(value)

    return orig_setitem(self, indexer, value)


def infer_fill_value(val, length: int):
    # GH#39044 we have to patch core.dtypes.missing.infer_fill_value
    #  to unwrap PandasArray bc it won't recognize PandasArray with
    #  is_extension_dtype
    if isinstance(val, PandasArray):
        val = val.to_numpy()

    return infer_fill_value_orig(val, length)


@pytest.fixture
def allow_in_pandas(monkeypatch):
    """
    A monkeypatch to tells pandas to let us in.

    By default, passing a PandasArray to an index / series / frame
    constructor will unbox that PandasArray to an ndarray, and treat
    it as a non-EA column. We don't want people using EAs without
    reason.

    The mechanism for this is a check against ABCPandasArray
    in each constructor.

    But, for testing, we need to allow them in pandas. So we patch
    the _typ of PandasArray, so that we evade the ABCPandasArray
    check.
    """
    with monkeypatch.context() as m:
        m.setattr(PandasArray, "_typ", "extension")
        m.setattr(pd.core.indexing, "infer_fill_value", infer_fill_value)
        m.setattr(pd.core.internals.Block, "setitem", setitem)
        yield


@pytest.fixture
def data(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return pd.Series([(i,) for i in range(100)]).array
    return PandasArray(np.arange(1, 101, dtype=dtype._dtype))


@pytest.fixture
def data_missing(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return PandasArray(np.array([np.nan, (1,)], dtype=object))
    return PandasArray(np.array([np.nan, 1.0]))


@pytest.fixture
def na_value():
    return np.nan


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return np.isnan(a) and np.isnan(b)

    return cmp


@pytest.fixture
def data_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    if dtype.numpy_dtype == "object":
        # Use an empty tuple for first element, then remove,
        # to disable np.array's shape inference.
        return PandasArray(np.array([(), (2,), (3,), (1,)], dtype=object)[1:])
    return PandasArray(np.array([1, 2, 0]))


@pytest.fixture
def data_missing_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    if dtype.numpy_dtype == "object":
        return PandasArray(np.array([(1,), np.nan, (0,)], dtype=object))
    return PandasArray(np.array([1, np.nan, 0]))


@pytest.fixture
def data_for_grouping(allow_in_pandas, dtype):
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    if dtype.numpy_dtype == "object":
        a, b, c = (1,), (2,), (3,)
    else:
        a, b, c = np.arange(3)
    return PandasArray(
        np.array([b, b, np.nan, np.nan, a, a, b, c], dtype=dtype.numpy_dtype)
    )


@pytest.fixture
def skip_numpy_object(dtype, request):
    """
    Tests for PandasArray with nested data. Users typically won't create
    these objects via `pd.array`, but they can show up through `.array`
    on a Series with nested data. Many of the base tests fail, as they aren't
    appropriate for nested data.

    This fixture allows these tests to be skipped when used as a usefixtures
    marker to either an individual test or a test class.
    """
    if dtype == "object":
        mark = pytest.mark.xfail(reason="Fails for object dtype")
        request.node.add_marker(mark)


skip_nested = pytest.mark.usefixtures("skip_numpy_object")


class BaseNumPyTests:
    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        # base class tests hard-code expected values with numpy dtypes,
        #  whereas we generally want the corresponding PandasDtype
        if (
            isinstance(right, pd.Series)
            and not isinstance(right.dtype, ExtensionDtype)
            and isinstance(left.dtype, PandasDtype)
        ):
            right = right.astype(PandasDtype(right.dtype))
        return tm.assert_series_equal(left, right, *args, **kwargs)


class TestCasting(BaseNumPyTests, base.BaseCastingTests):
    @skip_nested
    def test_astype_str(self, data):
        # ValueError: setting an array element with a sequence
        super().test_astype_str(data)


class TestConstructors(BaseNumPyTests, base.BaseConstructorsTests):
    @pytest.mark.skip(reason="We don't register our dtype")
    # We don't want to register. This test should probably be split in two.
    def test_from_dtype(self, data):
        pass

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # ValueError: Length of passed values is 1, index implies 3.
        super().test_series_constructor_scalar_with_index(data, dtype)


class TestDtype(BaseNumPyTests, base.BaseDtypeTests):
    @pytest.mark.skip(reason="Incorrect expected.")
    # we unsurprisingly clash with a NumPy name.
    def test_check_dtype(self, data):
        pass


class TestGetitem(BaseNumPyTests, base.BaseGetitemTests):
    @skip_nested
    def test_getitem_scalar(self, data):
        # AssertionError
        super().test_getitem_scalar(data)

    @skip_nested
    def test_take_series(self, data):
        # ValueError: PandasArray must be 1-dimensional.
        super().test_take_series(data)

    def test_loc_iloc_frame_single_dtype(self, data, request):
        npdtype = data.dtype.numpy_dtype
        if npdtype == object:
            # GH#33125
            mark = pytest.mark.xfail(
                reason="GH#33125 astype doesn't recognize data.dtype"
            )
            request.node.add_marker(mark)
        super().test_loc_iloc_frame_single_dtype(data)


class TestGroupby(BaseNumPyTests, base.BaseGroupbyTests):
    def test_groupby_extension_apply(
        self, data_for_grouping, groupby_apply_op, request
    ):
        dummy = groupby_apply_op([None])
        if (
            isinstance(dummy, pd.Series)
            and data_for_grouping.dtype.numpy_dtype == object
        ):
            mark = pytest.mark.xfail(reason="raises in MultiIndex construction")
            request.node.add_marker(mark)
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)


class TestInterface(BaseNumPyTests, base.BaseInterfaceTests):
    @skip_nested
    def test_array_interface(self, data):
        # NumPy array shape inference
        super().test_array_interface(data)


class TestMethods(BaseNumPyTests, base.BaseMethodsTests):
    @skip_nested
    def test_shift_fill_value(self, data):
        # np.array shape inference. Shift implementation fails.
        super().test_shift_fill_value(data)

    @skip_nested
    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        # Fails creating expected
        super().test_unique(data, box, method)

    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_frame(data_missing)

    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        # The "scalar" for this array isn't a scalar.
        super().test_fillna_copy_series(data_missing)

    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        # Test setup fails.
        super().test_searchsorted(data_for_sorting, as_series)

    @skip_nested
    def test_where_series(self, data, na_value, as_frame):
        # Test setup fails.
        super().test_where_series(data, na_value, as_frame)

    @pytest.mark.parametrize("repeats", [0, 1, 2, [1, 2, 3]])
    def test_repeat(self, data, repeats, as_series, use_numpy, request):
        if data.dtype.numpy_dtype == object and repeats != 0:
            mark = pytest.mark.xfail(reason="mask shapes mismatch")
            request.node.add_marker(mark)
        super().test_repeat(data, repeats, as_series, use_numpy)

    @pytest.mark.xfail(reason="PandasArray.diff may fail on dtype")
    def test_diff(self, data, periods):
        return super().test_diff(data, periods)

    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box, request):
        # Fails creating with _from_sequence
        if box is pd.DataFrame and data.dtype.numpy_dtype == object:
            mark = pytest.mark.xfail(reason="AssertionError in _get_same_shape_values")
            request.node.add_marker(mark)

        super().test_equals(data, na_value, as_series, box)


class TestArithmetics(BaseNumPyTests, base.BaseArithmeticOpsTests):
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    @skip_nested
    def test_divmod(self, data):
        super().test_divmod(data)

    @skip_nested
    def test_divmod_series_array(self, data):
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data, exc=None)

    @pytest.mark.skip("We implement ops")
    def test_error(self, data, all_arithmetic_operators):
        pass

    @skip_nested
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    @skip_nested
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    @skip_nested
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)


class TestPrinting(BaseNumPyTests, base.BasePrintingTests):
    pass


class TestNumericReduce(BaseNumPyTests, base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        result = getattr(s, op_name)(skipna=skipna)
        # avoid coercing int -> float. Just cast to the actual numpy type.
        expected = getattr(s.astype(s.dtype._dtype), op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_boolean_reductions, skipna):
        super().test_reduce_series(data, all_boolean_reductions, skipna)


@skip_nested
class TestBooleanReduce(BaseNumPyTests, base.BaseBooleanReduceTests):
    pass


class TestMissing(BaseNumPyTests, base.BaseMissingTests):
    @skip_nested
    def test_fillna_scalar(self, data_missing):
        # Non-scalar "scalar" values.
        super().test_fillna_scalar(data_missing)

    @skip_nested
    def test_fillna_series_method(self, data_missing, fillna_method):
        # Non-scalar "scalar" values.
        super().test_fillna_series_method(data_missing, fillna_method)

    @skip_nested
    def test_fillna_series(self, data_missing):
        # Non-scalar "scalar" values.
        super().test_fillna_series(data_missing)

    @skip_nested
    def test_fillna_frame(self, data_missing):
        # Non-scalar "scalar" values.
        super().test_fillna_frame(data_missing)

    def test_fillna_fill_other(self, data_missing):
        # Same as the parent class test, but with PandasDtype for expected["B"]
        #  instead of equivalent numpy dtype
        data = data_missing
        result = pd.DataFrame({"A": data, "B": [np.nan] * len(data)}).fillna({"B": 0.0})

        expected = pd.DataFrame({"A": data, "B": [0.0] * len(result)})
        expected["B"] = expected["B"].astype(PandasDtype(expected["B"].dtype))

        self.assert_frame_equal(result, expected)


class TestReshaping(BaseNumPyTests, base.BaseReshapingTests):
    @skip_nested
    def test_merge(self, data, na_value):
        # Fails creating expected
        super().test_merge(data, na_value)

    @skip_nested
    def test_merge_on_extension_array(self, data):
        # Fails creating expected
        super().test_merge_on_extension_array(data)

    @skip_nested
    def test_merge_on_extension_array_duplicates(self, data):
        # Fails creating expected
        super().test_merge_on_extension_array_duplicates(data)

    @skip_nested
    def test_transpose_frame(self, data):
        super().test_transpose_frame(data)


class TestSetitem(BaseNumPyTests, base.BaseSetitemTests):
    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    @skip_nested
    def test_setitem_loc_scalar_mixed(self, data):
        # AssertionError
        super().test_setitem_loc_scalar_mixed(data)

    @skip_nested
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        # AssertionError
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @skip_nested
    def test_setitem_iloc_scalar_mixed(self, data):
        # AssertionError
        super().test_setitem_iloc_scalar_mixed(data)

    @skip_nested
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        # AssertionError
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)

    @skip_nested
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # ValueError: cannot set using a list-like indexer with a different
        # length than the value
        super().test_setitem_mask_broadcast(data, setter)

    @skip_nested
    def test_setitem_scalar_key_sequence_raise(self, data):
        # Failed: DID NOT RAISE <class 'ValueError'>
        super().test_setitem_scalar_key_sequence_raise(data)

    # TODO: there is some issue with PandasArray, therefore,
    #   skip the setitem test for now, and fix it later (GH 31446)

    @skip_nested
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        super().test_setitem_mask(data, mask, box_in_series)

    def test_setitem_mask_raises(self, data, box_in_series):
        super().test_setitem_mask_raises(data, box_in_series)

    @skip_nested
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        super().test_setitem_integer_array(data, idx, box_in_series)

    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),
            pytest.param([0, 1, 2, pd.NA], True, marks=pytest.mark.xfail),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],
    )
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        super().test_setitem_slice(data, box_in_series)

    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        super().test_setitem_loc_iloc_slice(data)

    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer, request):
        # https://github.com/pandas-dev/pandas/issues/32395
        df = pd.DataFrame({"data": pd.Series(data)})
        result = pd.DataFrame(index=df.index)

        key = full_indexer(df)
        result.loc[key, "data"] = df["data"]._values

        expected = pd.DataFrame({"data": data})
        if data.dtype.numpy_dtype != object:
            # For PandasArray we expect to get unboxed to numpy
            expected = pd.DataFrame({"data": data.to_numpy()})

        if isinstance(key, slice) and (
            key == slice(None) and data.dtype.numpy_dtype != object
        ):
            mark = pytest.mark.xfail(
                reason="This case goes through a different code path"
            )
            # Other cases go through Block.setitem
            request.node.add_marker(mark)

        self.assert_frame_equal(result, expected)

    def test_setitem_series(self, data, full_indexer):
        # https://github.com/pandas-dev/pandas/issues/32395
        ser = pd.Series(data, name="data")
        result = pd.Series(index=ser.index, dtype=object, name="data")

        key = full_indexer(ser)
        result.loc[key] = ser

        # For PandasArray we expect to get unboxed to numpy
        expected = pd.Series(data.to_numpy(), name="data")
        self.assert_series_equal(result, expected)


@skip_nested
class TestParsing(BaseNumPyTests, base.BaseParsingTests):
    pass

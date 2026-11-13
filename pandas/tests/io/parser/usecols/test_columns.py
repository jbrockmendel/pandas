"""
Tests for the ``columns`` keyword and ``pd.columns`` selections.
"""

from io import (
    BytesIO,
    StringIO,
)

import pytest

import pandas as pd
import pandas._testing as tm

DATA = "a,b,c,d\n1,2,3,4\n5,6,7,8\n"


@pytest.mark.parametrize(
    "columns, expected_cols",
    [
        (["c", "a"], ["c", "a"]),
        (pd.columns[[2, 0]], ["c", "a"]),
        (pd.columns[[-1, 0]], ["d", "a"]),
        (pd.columns[1:], ["b", "c", "d"]),
        (pd.columns[::-1][:2], ["d", "c"]),
        (pd.columns[[True, False, False, True]], ["a", "d"]),
        (lambda cols: cols.drop("b"), ["a", "c", "d"]),
        (lambda cols: cols.isin(["d", "a"]), ["a", "d"]),
        (lambda cols: cols[[3, 1]], ["d", "b"]),
    ],
)
def test_columns_selection_order(all_parsers, columns, expected_cols):
    parser = all_parsers
    result = parser.read_csv(StringIO(DATA), columns=columns)
    expected = parser.read_csv(StringIO(DATA))[expected_cols]
    tm.assert_frame_equal(result, expected)


def test_columns_empty(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO(DATA), columns=[])
    assert list(result.columns) == []


@pytest.mark.parametrize("index_col", [0, "c"])
def test_columns_index_col_refers_to_selection(all_parsers, index_col):
    parser = all_parsers
    result = parser.read_csv(
        StringIO(DATA), columns=pd.columns[[2, 0]], index_col=index_col
    )
    expected = pd.DataFrame({"a": [1, 5]}, index=pd.Index([3, 7], name="c"))
    tm.assert_frame_equal(result, expected)


def test_columns_header_none(all_parsers):
    # positions and the default integer labels agree; the index stays unnamed
    parser = all_parsers
    by_label = parser.read_csv(StringIO(DATA), header=None, columns=[3, 1])
    by_position = parser.read_csv(
        StringIO(DATA), header=None, columns=pd.columns[[3, 1]]
    )
    expected = pd.DataFrame({3: ["d", "4", "8"], 1: ["b", "2", "6"]})
    tm.assert_frame_equal(by_label, expected)
    tm.assert_frame_equal(by_position, expected)

    result = parser.read_csv(StringIO(DATA), header=None, columns=[3, 1], index_col=0)
    expected = pd.DataFrame({1: ["b", "2", "6"]}, index=pd.Index(["d", "4", "8"]))
    tm.assert_frame_equal(result, expected)


def test_columns_select_from_names(all_parsers):
    # names replaces the header, so labels and positions refer to it
    parser = all_parsers
    result = parser.read_csv(
        StringIO(DATA), names=list("wxyz"), header=0, columns=["z", "w"]
    )
    expected = pd.DataFrame({"z": [4, 8], "w": [1, 5]})
    tm.assert_frame_equal(result, expected)


def test_columns_short_names_raises(all_parsers):
    parser = all_parsers
    msg = "'names' replaces the whole header and must name every column"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(
            StringIO(DATA), names=["p", "q"], header=0, columns=pd.columns[[1, 0]]
        )


def test_columns_names_wider_than_data(all_parsers, request):
    # names longer than the rows still sets the row width for ragged files
    parser = all_parsers
    if parser.engine == "pyarrow":
        request.applymarker(
            pytest.mark.xfail(reason="pyarrow rejects ragged rows", strict=True)
        )
    data = "1,2\n3,4,5\n6\n"
    result = parser.read_csv(
        StringIO(data), header=None, names=["x", "y", "z"], columns=pd.columns[[2, 0]]
    )
    expected = pd.DataFrame({"z": [None, 5.0, None], "x": [1, 3, 6]})
    tm.assert_frame_equal(result, expected)


def test_columns_dtype_by_label(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO(DATA), columns=["c", "a"], dtype={"a": "float64"})
    expected = pd.DataFrame({"c": [3, 7], "a": [1.0, 5.0]})
    tm.assert_frame_equal(result, expected)


def test_columns_chunksize(c_parser_only):
    parser = c_parser_only
    with parser.read_csv(
        StringIO(DATA), columns=pd.columns[[2, 0]], chunksize=1
    ) as reader:
        result = pd.concat(reader)
    expected = pd.DataFrame({"c": [3, 7], "a": [1, 5]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs, error, msg",
    [
        ({"columns": ["c", "zz"]}, ValueError, r"not found in the header: \['zz'\]"),
        ({"columns": pd.columns[[7]]}, IndexError, "out of bounds"),
        ({"columns": ["a", "a"]}, ValueError, "same column more than once"),
        ({"columns": ["a"], "usecols": ["a"]}, ValueError, "only one of"),
        ({"columns": "a"}, TypeError, "'columns' must be list-like"),
    ],
)
def test_columns_invalid(all_parsers, kwargs, error, msg):
    parser = all_parsers
    with pytest.raises(error, match=msg):
        parser.read_csv(StringIO(DATA), **kwargs)


def test_columns_multirow_header_raises(python_parser_only):
    parser = python_parser_only
    with pytest.raises(ValueError, match="multi-row 'header'"):
        parser.read_csv(StringIO(DATA), header=[0, 1], columns=["a"])


@pytest.mark.parametrize("key", ["a", ["a", "b"], slice("a", "c"), 1.5])
def test_column_selection_rejects_labels(key):
    with pytest.raises(TypeError, match="selects by position"):
        pd.columns[key]


def test_column_selection_repr():
    assert repr(pd.columns[[2, 0]]) == "pd.columns[[2, 0]]"
    assert repr(pd.columns[1:][::2]) == "pd.columns[1:][::2]"
    assert repr(pd.columns[-1]) == "pd.columns[-1]"


def test_read_excel_columns():
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]})
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")

    buf.seek(0)
    result = pd.read_excel(buf, engine="openpyxl", columns=pd.columns[[2, 0]])
    tm.assert_frame_equal(result, df[["c", "a"]])

    buf.seek(0)
    result = pd.read_excel(buf, engine="openpyxl", columns=["d", "b"], index_col="b")
    tm.assert_frame_equal(result, df[["d", "b"]].set_index("b"))

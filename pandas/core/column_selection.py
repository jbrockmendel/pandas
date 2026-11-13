"""
Deferred column selection for the ``columns`` keyword of the IO readers.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.util._decorators import set_module

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_list_like,
)

from pandas.core.indexes.api import (
    Index,
    RangeIndex,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Sequence,
    )

    from pandas._typing import npt

    from pandas import DataFrame


@set_module("pandas.api.typing")
class ColumnSelection:
    """
    Deferred positional selection from the columns of a DataFrame being read.

    Created by indexing :data:`pandas.columns`, which stands for the
    ``.columns`` of the DataFrame a reader is about to produce. Indexing works
    as it does on a real ``df.columns``: by position, with an integer, a slice,
    a list of integers or a boolean mask. The selection is resolved once the
    reader knows the header.

    This is not meant to be instantiated directly; index ``pd.columns`` instead.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_excel : Read an Excel file into a DataFrame.

    Examples
    --------
    >>> pd.columns[[2, 0]]
    pd.columns[[2, 0]]
    >>> pd.columns[1:][::2]
    pd.columns[1:][::2]
    """

    def __init__(self, keys: tuple[Any, ...] = ()) -> None:
        self._keys = keys

    def __getitem__(self, key: Any) -> ColumnSelection:
        return ColumnSelection((*self._keys, _validate_key(key)))

    def __repr__(self) -> str:
        return "pd.columns" + "".join(f"[{_format_key(key)}]" for key in self._keys)

    def _positions(self, ncols: int) -> npt.NDArray[np.intp]:
        positions = np.arange(ncols, dtype=np.intp)
        for key in self._keys:
            positions = np.atleast_1d(positions[key])
        return positions


#: Stand-in for the ``.columns`` of the DataFrame a reader will produce.
columns = ColumnSelection()
columns.__module__ = "pandas"


def _validate_key(key: Any) -> Any:
    if isinstance(key, slice):
        for part in (key.start, key.stop, key.step):
            if part is not None and not is_integer(part):
                raise TypeError(
                    "pd.columns selects by position, so slices must be "
                    "integers; select columns "
                    "by label by passing the labels directly, e.g. "
                    "columns=['a', 'b']."
                )
        return key
    if is_integer(key):
        return int(key)
    if is_list_like(key):
        arr = np.asarray(key)
        if len(arr) == 0:
            return arr.astype(np.intp)
        if arr.dtype.kind in "iu":
            return arr.astype(np.intp)
        if arr.dtype.kind == "b":
            return arr
    raise TypeError(
        "pd.columns selects by position and accepts an integer, an integer "
        "slice, a list of integers or a boolean mask; select columns by label "
        f"by passing the labels directly, e.g. columns=['a', 'b']. Got {key!r}."
    )


def _format_key(key: Any) -> str:
    if isinstance(key, slice):
        parts = ["" if part is None else str(part) for part in (key.start, key.stop)]
        text = ":".join(parts)
        return text if key.step is None else f"{text}:{key.step}"
    if isinstance(key, np.ndarray):
        return repr(key.tolist())
    return repr(key)


def resolve_columns(
    selection: ColumnSelection | Callable[[Index], Any] | Any,
    header: Index,
) -> npt.NDArray[np.intp]:
    """
    Return the positions in ``header`` that ``selection`` picks, in its order.
    """
    if isinstance(selection, ColumnSelection):
        try:
            positions = selection._positions(len(header))
        except IndexError as err:
            raise IndexError(
                f"{selection!r} is out of bounds for a header with "
                f"{len(header)} columns"
            ) from err
    else:
        if callable(selection):
            selection = selection(header)
        if not is_list_like(selection):
            raise TypeError(
                "'columns' must be list-like, a callable, or a pd.columns "
                f"selection, got {type(selection).__name__}"
            )
        if is_bool_dtype(selection) and len(selection) == len(header):
            positions = np.flatnonzero(np.asarray(selection))
        else:
            labels = Index(list(selection), tupleize_cols=False)
            positions = header.get_indexer(labels)
            missing = labels[positions == -1]
            if len(missing):
                raise ValueError(f"columns not found in the header: {list(missing)}")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("'columns' selects the same column more than once")
    return positions.astype(np.intp, copy=False)


SHORT_NAMES_MSG = (
    "With 'columns', 'names' replaces the whole header and must name every "
    "column in the file; rename the result to relabel a selection."
)


class ColumnsResolver:
    """
    Carry a ``columns`` selection through a text-parser engine.

    The C and python engines and the parallel C reader receive this in place
    of ``usecols`` and resolve it to positions once the header is known.
    ``finalize`` then puts the parsed columns in the requested order and sets
    the index, so ``index_col`` refers to the selected columns.
    """

    def __init__(
        self,
        selection: Any,
        index_col: Hashable | list[Hashable] | bool | None,
        unnamed_header: bool,
    ) -> None:
        self.selection = selection
        self.index_col = index_col
        # header=None and no names: the index gets no name, as with index_col
        self.unnamed_header = unnamed_header
        self.positions: npt.NDArray[np.intp] | None = None

    def resolve(self, names: Sequence[Hashable]) -> set[int]:
        header = Index(list(names), tupleize_cols=False)
        self.positions = resolve_columns(self.selection, header)
        return set(self.positions.tolist())

    def __call__(self, name: Hashable) -> bool:
        # The engines must call ``resolve``; a per-name call would mean some
        # code path treated this as an ordinary callable ``usecols``.
        raise AssertionError("ColumnsResolver must be resolved, not called")

    def finalize(self, frame: DataFrame) -> DataFrame:
        if self.positions is None:
            # the engine read every column (pyarrow)
            if not isinstance(frame.index, RangeIndex):
                # a short 'names' made the engine infer an index
                raise ValueError(SHORT_NAMES_MSG)
            self.resolve(list(frame.columns))
            assert self.positions is not None
            frame = frame.iloc[:, self.positions]
        else:
            # the engine returned the selected columns in file order
            order = np.searchsorted(np.sort(self.positions), self.positions)
            if not (order == np.arange(len(order))).all():
                frame = frame.iloc[:, order]

        index_col = self.index_col
        if index_col is None or index_col is False:
            return frame
        if not isinstance(index_col, list):
            index_col = [index_col]
        keys = [frame.columns[key] if is_integer(key) else key for key in index_col]
        frame = frame.set_index(keys)
        if self.unnamed_header:
            frame.index.names = [None] * frame.index.nlevels
        return frame

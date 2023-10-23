import calendar
from datetime import datetime
import locale
import unicodedata

from dateutil.tz import tzlocal
from hypothesis import (
    given,
    strategies as st,
)
import numpy as np
import pytest

from pandas._libs.tslibs import timezones
from pandas.compat import IS64

from pandas import (
    NaT,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm

from pandas.tseries.frequencies import to_offset


class TestTimestampFields:
    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # GH#21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)
        dt = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)

        result = stamp.timetz()
        expected = dt.timetz()

        assert result == expected

    def test_properties_business(self):
        freq = to_offset("B")

        ts = Timestamp("2017-10-01")
        assert ts.dayofweek == 6
        assert ts.day_of_week == 6
        assert ts.is_month_start  # not a weekday
        assert not freq.is_month_start(ts)
        assert freq.is_month_start(ts + Timedelta(days=1))
        assert not freq.is_quarter_start(ts)
        assert freq.is_quarter_start(ts + Timedelta(days=1))

        ts = Timestamp("2017-09-30")
        assert ts.dayofweek == 5
        assert ts.day_of_week == 5
        assert ts.is_month_end
        assert not freq.is_month_end(ts)
        assert freq.is_month_end(ts - Timedelta(days=1))
        assert ts.is_quarter_end
        assert not freq.is_quarter_end(ts)
        assert freq.is_quarter_end(ts - Timedelta(days=1))

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ["year", 2014],
            ["month", 12],
            ["day", 31],
            ["hour", 23],
            ["minute", 59],
            ["second", 0],
            ["microsecond", 0],
            ["nanosecond", 0],
            ["dayofweek", 2],
            ["day_of_week", 2],
            ["quarter", 4],
            ["dayofyear", 365],
            ["day_of_year", 365],
            ["week", 1],
            ["daysinmonth", 31],
        ],
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_fields(self, attr, expected, tz):
        # GH#10050
        # GH#13303
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        result = getattr(ts, attr)
        # that we are int like
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_millisecond_raises(self, tz):
        ts = Timestamp("2014-12-31 23:59:00", tz=tz)
        msg = "'Timestamp' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            ts.millisecond

    @pytest.mark.parametrize(
        "start", ["is_month_start", "is_quarter_start", "is_year_start"]
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_start(self, start, tz):
        ts = Timestamp("2014-01-01 00:00:00", tz=tz)
        assert getattr(ts, start)

    @pytest.mark.parametrize("end", ["is_month_end", "is_year_end", "is_quarter_end"])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_is_end(self, end, tz):
        ts = Timestamp("2014-12-31 23:59:59", tz=tz)
        assert getattr(ts, end)

    # GH#12806
    @pytest.mark.parametrize(
        "data",
        [Timestamp("2017-08-28 23:00:00"), Timestamp("2017-08-28 23:00:00", tz="EST")],
    )
    # error: Unsupported operand types for + ("List[None]" and "List[str]")
    @pytest.mark.parametrize(
        "time_locale", [None] + tm.get_locales()  # type: ignore[operator]
    )
    def test_names(self, data, time_locale):
        # GH#17354
        # Test .day_name(), .month_name
        if time_locale is None:
            expected_day = "Monday"
            expected_month = "August"
        else:
            with tm.set_locale(time_locale, locale.LC_TIME):
                expected_day = calendar.day_name[0].capitalize()
                expected_month = calendar.month_name[8].capitalize()

        result_day = data.day_name(time_locale)
        result_month = data.month_name(time_locale)

        # Work around https://github.com/pandas-dev/pandas/issues/22342
        # different normalizations
        expected_day = unicodedata.normalize("NFD", expected_day)
        expected_month = unicodedata.normalize("NFD", expected_month)

        result_day = unicodedata.normalize("NFD", result_day)
        result_month = unicodedata.normalize("NFD", result_month)

        assert result_day == expected_day
        assert result_month == expected_month

        # Test NaT
        nan_ts = Timestamp(NaT)
        assert np.isnan(nan_ts.day_name(time_locale))
        assert np.isnan(nan_ts.month_name(time_locale))

    def test_is_leap_year(self, tz_naive_fixture):
        tz = tz_naive_fixture
        if not IS64 and tz == tzlocal():
            # https://github.com/dateutil/dateutil/issues/197
            pytest.skip(
                "tzlocal() on a 32 bit platform causes internal overflow errors"
            )
        # GH#13727
        dt = Timestamp("2000-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year
        assert isinstance(dt.is_leap_year, bool)

        dt = Timestamp("1999-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

        dt = Timestamp("2004-01-01 00:00:00", tz=tz)
        assert dt.is_leap_year

        dt = Timestamp("2100-01-01 00:00:00", tz=tz)
        assert not dt.is_leap_year

    def test_woy_boundary(self):
        # make sure weeks at year boundaries are correct
        d = datetime(2013, 12, 31)
        result = Timestamp(d).week
        expected = 1  # ISO standard
        assert result == expected

        d = datetime(2008, 12, 28)
        result = Timestamp(d).week
        expected = 52  # ISO standard
        assert result == expected

        d = datetime(2009, 12, 31)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 1)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        d = datetime(2010, 1, 3)
        result = Timestamp(d).week
        expected = 53  # ISO standard
        assert result == expected

        result = np.array(
            [
                Timestamp(datetime(*args)).week
                for args in [(2000, 1, 1), (2000, 1, 2), (2005, 1, 1), (2005, 1, 2)]
            ]
        )
        assert (result == [52, 52, 53, 53]).all()

    def test_resolution(self):
        # GH#21336, GH#21365
        dt = Timestamp("2100-01-01 00:00:00.000000000")
        assert dt.resolution == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib datetime behavior
        assert Timestamp.resolution == Timedelta(nanoseconds=1)

        assert dt.as_unit("us").resolution == Timedelta(microseconds=1)
        assert dt.as_unit("ms").resolution == Timedelta(milliseconds=1)
        assert dt.as_unit("s").resolution == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "date_string, expected",
        [
            ("0000-2-29", 1),
            ("0000-3-1", 2),
            ("1582-10-14", 3),
            ("-0040-1-1", 4),
            ("2023-06-18", 6),
        ],
    )
    def test_dow_historic(self, date_string, expected):
        # GH#53738
        ts = Timestamp(date_string)
        dow = ts.weekday()
        assert dow == expected

    @given(
        ts=st.datetimes(),
        sign=st.sampled_from(["-", ""]),
    )
    def test_dow_parametric(self, ts, sign):
        # GH#53738
        ts = (
            f"{sign}{str(ts.year).zfill(4)}"
            f"-{str(ts.month).zfill(2)}"
            f"-{str(ts.day).zfill(2)}"
        )
        result = Timestamp(ts).weekday()
        expected = (
            (np.datetime64(ts) - np.datetime64("1970-01-01")).astype("int64") - 4
        ) % 7
        assert result == expected

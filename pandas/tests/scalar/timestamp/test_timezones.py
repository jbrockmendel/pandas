"""
Tests for Timestamp timezone-related methods
"""
from datetime import (
    datetime,
    timezone,
)

import dateutil
from dateutil.tz import tzoffset
import pytest

from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td

from pandas import Timestamp

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Cannot assign to a type
    ZoneInfo = None  # type: ignore[misc, assignment]


class TestTimestampTZOperations:
    # ------------------------------------------------------------------
    # Timestamp.tz_convert

    @pytest.mark.parametrize(
        "stamp",
        [
            "2014-02-01 09:00",
            "2014-07-08 09:00",
            "2014-11-01 17:00",
            "2014-11-05 00:00",
        ],
    )
    def test_tz_convert_roundtrip(self, stamp, tz_aware_fixture):
        tz = tz_aware_fixture

        ts = Timestamp(stamp, tz="UTC")
        converted = ts.tz_convert(tz)

        reset = converted.tz_convert(None)
        assert reset == Timestamp(stamp)
        assert reset.tzinfo is None
        assert reset == converted.tz_convert("UTC").tz_localize(None)

    @pytest.mark.parametrize("tzstr", ["US/Eastern", "dateutil/US/Eastern"])
    def test_astimezone(self, tzstr):
        # astimezone is an alias for tz_convert, so keep it with
        # the tz_convert tests
        utcdate = Timestamp("3/11/2012 22:00", tz="UTC")
        expected = utcdate.tz_convert(tzstr)
        result = utcdate.astimezone(tzstr)
        assert expected == result
        assert isinstance(result, Timestamp)

    @td.skip_if_windows
    def test_tz_convert_utc_with_system_utc(self):
        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

        # from system utc to real utc
        ts = Timestamp("2001-01-05 11:56", tz=timezones.maybe_get_tz("dateutil/UTC"))
        # check that the time hasn't changed.
        assert ts == ts.tz_convert(dateutil.tz.tzutc())

    # ------------------------------------------------------------------
    # Timestamp.__init__ with tz str or tzinfo

    def test_timestamp_constructor_tz_utc(self):
        utc_stamp = Timestamp("3/11/2012 05:00", tz="utc")
        assert utc_stamp.tzinfo is timezone.utc
        assert utc_stamp.hour == 5

        utc_stamp = Timestamp("3/11/2012 05:00").tz_localize("utc")
        assert utc_stamp.hour == 5

    def test_timestamp_to_pydatetime_tzoffset(self):
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp("3/11/2012 04:00", tz=tzinfo)
        result = Timestamp(expected.to_pydatetime())
        assert expected == result

    def test_timestamp_timetz_equivalent_with_datetime_tz(self, tz_naive_fixture):
        # GH21358
        tz = timezones.maybe_get_tz(tz_naive_fixture)

        stamp = Timestamp("2018-06-04 10:20:30", tz=tz)
        _datetime = datetime(2018, 6, 4, hour=10, minute=20, second=30, tzinfo=tz)

        result = stamp.timetz()
        expected = _datetime.timetz()

        assert result == expected

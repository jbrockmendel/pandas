from dateutil.tz import tzoffset

from pandas import Timestamp


class TestToPyDatetime:
    def test_timestamp_to_pydatetime_tzoffset(self):
        tzinfo = tzoffset(None, 7200)
        expected = Timestamp("3/11/2012 04:00", tz=tzinfo)
        result = Timestamp(expected.to_pydatetime())
        assert expected == result

import pytest

from src.psx_predictor.scraper.psx_dps_index_scraper import PsxDpsIndexScraper

_SAMPLE_INDICES_HTML = """
<html><body>
<table>
<tr><th>Index</th><th>High</th><th>Low</th><th>Current</th><th>Change</th><th>% Change</th></tr>
<tr><td>KSE100</td><td>166,070.47</td><td>164,773.66</td><td>164,776.44</td><td>-347.97</td><td>-0.21%</td></tr>
<tr><td>KSE100PR</td><td>49,811.49</td><td>49,422.52</td><td>49,423.36</td><td>-104.37</td><td>-0.21%</td></tr>
<tr><td>ALLSHR</td><td>100,730.93</td><td>99,992.16</td><td>99,975.85</td><td>-181.84</td><td>-0.18%</td></tr>
<tr><td>KSE30</td><td>49,288.80</td><td>48,846.36</td><td>48,868.27</td><td>-135.88</td><td>-0.28%</td></tr>
<tr><td>KMI30</td><td>239,271.63</td><td>237,403.18</td><td>237,568.77</td><td>-245.20</td><td>-0.10%</td></tr>
<tr><td>BKTI</td><td>45,500.00</td><td>45,300.00</td><td>45,396.53</td><td>-125.46</td><td>-0.28%</td></tr>
<tr><td>OGTI</td><td>33,500.00</td><td>33,300.00</td><td>33,428.39</td><td>-129.64</td><td>-0.39%</td></tr>
<tr><td>SOME_OTHER_INDEX</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.00%</td></tr>
</table>
<table><tr><td>Sector Indices</td></tr></table>
</body></html>
"""


def _mock_response(mocker, status_code=200, text=_SAMPLE_INDICES_HTML):
    resp = mocker.Mock()
    resp.status_code = status_code
    resp.text = text
    return resp


class TestFetchLiveSnapshot:
    def test_parses_all_tracked_indices(self, mocker):
        mocker.patch("requests.get", return_value=_mock_response(mocker))
        snapshot = PsxDpsIndexScraper().fetch_live_snapshot()

        assert snapshot["KSE100"] == pytest.approx((164776.44, -0.0021))
        assert snapshot["KMI30"] == pytest.approx((237568.77, -0.0010))
        assert snapshot["KSE30"] == pytest.approx((48868.27, -0.0028))

    def test_ignores_untracked_indices(self, mocker):
        mocker.patch("requests.get", return_value=_mock_response(mocker))
        snapshot = PsxDpsIndexScraper().fetch_live_snapshot()
        assert "SOME_OTHER_INDEX" not in snapshot

    def test_returns_empty_dict_on_non_200(self, mocker):
        mocker.patch("requests.get", return_value=_mock_response(mocker, status_code=403))
        assert PsxDpsIndexScraper().fetch_live_snapshot() == {}

    def test_returns_empty_dict_on_request_exception(self, mocker):
        mocker.patch("requests.get", side_effect=Exception("connection refused"))
        assert PsxDpsIndexScraper().fetch_live_snapshot() == {}

    def test_returns_empty_dict_when_no_table_present(self, mocker):
        mocker.patch("requests.get", return_value=_mock_response(mocker, text="<html><body>no data</body></html>"))
        assert PsxDpsIndexScraper().fetch_live_snapshot() == {}


class TestSyncAllIndices:
    def test_upserts_one_row_for_today_with_all_columns(self, mocker):
        mocker.patch.object(
            PsxDpsIndexScraper, "fetch_live_snapshot",
            return_value={"KSE100": (164776.44, -0.0021), "KMI30": (237568.77, -0.0010)},
        )
        mock_upsert = mocker.patch("src.psx_predictor.scraper.psx_dps_index_scraper.upsert_macro_indicators")

        result = PsxDpsIndexScraper().sync_all_indices()

        assert result is True
        mock_upsert.assert_called_once()
        df = mock_upsert.call_args[0][0]
        assert len(df) == 1
        assert df.iloc[0]["kse100_real_index_level"] == 164776.44
        assert df.iloc[0]["kse100_real_return_pct"] == -0.0021
        assert df.iloc[0]["kmi30_index_level"] == 237568.77

    def test_returns_false_when_snapshot_is_empty(self, mocker):
        mocker.patch.object(PsxDpsIndexScraper, "fetch_live_snapshot", return_value={})
        mock_upsert = mocker.patch("src.psx_predictor.scraper.psx_dps_index_scraper.upsert_macro_indicators")

        result = PsxDpsIndexScraper().sync_all_indices()

        assert result is False
        mock_upsert.assert_not_called()

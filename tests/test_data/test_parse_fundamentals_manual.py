import time

import pandas as pd
import pytest

import src.psx_predictor.data.parse_fundamentals_manual as pfm


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, mocker):
    """Every test gets its own cache/lock file instead of touching the real
    project's data/cache/fundamentals_pdf_cache.json."""
    mocker.patch.object(pfm, "_CACHE_PATH", tmp_path / "cache.json")
    mocker.patch.object(pfm, "_LOCK_PATH", tmp_path / "cache.lock")


def _write_long_format_csv(path, report_date="2024-12-31", revenue=50000.0):
    path.write_text(
        "date,revenue,net_income\n"
        f"{report_date},{revenue},10000.0\n"
    )


class TestCaching:
    def test_first_parse_populates_cache_and_returns_data(self, tmp_path, mocker):
        f = tmp_path / "TEST_2024_Annual.csv"
        _write_long_format_csv(f)
        mocker.patch.object(pfm, "inspect_and_log_all", return_value=[f])

        df = pfm.parse_fundamentals_manual("fundamentals_test")

        assert not df.empty
        assert df.iloc[0]["revenue"] == 50000.0
        cache = pfm._load_cache()
        assert f.name in cache["fundamentals_test"]

    def test_second_call_on_unchanged_file_skips_reparse(self, tmp_path, mocker):
        f = tmp_path / "TEST_2024_Annual.csv"
        _write_long_format_csv(f)
        mocker.patch.object(pfm, "inspect_and_log_all", return_value=[f])

        pfm.parse_fundamentals_manual("fundamentals_test")  # cold
        spy = mocker.spy(pd, "read_csv")
        df = pfm.parse_fundamentals_manual("fundamentals_test")  # warm

        spy.assert_not_called()
        assert df.empty  # nothing new/changed to return

    def test_modified_file_is_reparsed(self, tmp_path, mocker):
        f = tmp_path / "TEST_2024_Annual.csv"
        _write_long_format_csv(f, revenue=50000.0)
        mocker.patch.object(pfm, "inspect_and_log_all", return_value=[f])
        pfm.parse_fundamentals_manual("fundamentals_test")

        time.sleep(0.05)  # ensure a distinguishable mtime
        _write_long_format_csv(f, revenue=99999.0)
        df = pfm.parse_fundamentals_manual("fundamentals_test")

        assert not df.empty
        assert df.iloc[0]["revenue"] == 99999.0

    def test_different_sources_have_independent_cache_entries(self, tmp_path, mocker):
        f1 = tmp_path / "PSO_2024_Annual.csv"
        f2 = tmp_path / "MEBL_2024_Annual.csv"
        _write_long_format_csv(f1)
        _write_long_format_csv(f2)

        mocker.patch.object(pfm, "inspect_and_log_all", side_effect=lambda source: [f1] if "pso" in source else [f2])

        pfm.parse_fundamentals_manual("fundamentals_pso")
        pfm.parse_fundamentals_manual("fundamentals_mebl")

        cache = pfm._load_cache()
        assert f1.name in cache["fundamentals_pso"]
        assert f2.name in cache["fundamentals_mebl"]
        assert f1.name not in cache.get("fundamentals_mebl", {})


class TestPdfTimeoutGuard:
    def test_slow_pdf_is_left_uncached_for_retry(self, tmp_path, mocker):
        mocker.patch.object(pfm, "_PDF_PARSE_TIMEOUT_SECONDS", 0.1)
        mocker.patch.object(pfm, "_extract_pdf_records", side_effect=lambda *a, **kw: time.sleep(2) or [])

        f = tmp_path / "TEST_2024_Annual.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        mocker.patch.object(pfm, "inspect_and_log_all", return_value=[f])

        start = time.monotonic()
        df = pfm.parse_fundamentals_manual("fundamentals_test")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0  # doesn't block on the slow worker thread
        assert df.empty
        cache = pfm._load_cache()
        assert f.name not in cache.get("fundamentals_test", {})

    def test_fast_pdf_is_cached_normally(self, tmp_path, mocker):
        mocker.patch.object(pfm, "_PDF_PARSE_TIMEOUT_SECONDS", 5)
        mocker.patch.object(
            pfm, "_extract_pdf_records",
            return_value=[{"ticker": "TEST", "report_date": pd.to_datetime("2024-12-31").date(), "metric": "revenue", "value": 50000.0}],
        )

        f = tmp_path / "TEST_2024_Annual.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        mocker.patch.object(pfm, "inspect_and_log_all", return_value=[f])

        df = pfm.parse_fundamentals_manual("fundamentals_test")

        assert not df.empty
        cache = pfm._load_cache()
        assert f.name in cache["fundamentals_test"]


class TestUpsertFundamentalsPlausibilityGate:
    def test_rejects_implausibly_small_currency_values(self, mocker):
        mock_conn = mocker.MagicMock()
        mock_engine = mocker.MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.connect.return_value.__exit__.return_value = False
        mocker.patch.object(pfm, "engine", mock_engine)

        # Real bug this guards against: total_debt=17.0 identically across
        # many quarters is a misread ratio/wrong-row, not a real PKR value.
        df = pd.DataFrame([{
            "ticker": "TEST", "report_date": pd.to_datetime("2024-12-31").date(),
            "revenue": 5.0, "net_income": 50000.0,
        }])

        pfm.upsert_fundamentals(df)

        executed_params = mock_conn.execute.call_args[0][1]
        assert "revenue" not in executed_params  # dropped as implausible
        assert executed_params["net_income"] == 50000.0

    def test_skips_db_call_for_empty_dataframe(self, mocker):
        mock_engine = mocker.MagicMock()
        mocker.patch.object(pfm, "engine", mock_engine)
        pfm.upsert_fundamentals(pd.DataFrame())
        mock_engine.connect.assert_not_called()

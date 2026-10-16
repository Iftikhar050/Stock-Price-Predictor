import pandas as pd

from src.psx_predictor.pipelines.orchestrator import run_full_data_pipeline


def _patch_all_subpipelines(mocker):
    """Every sub-pipeline orchestrator.py calls, mocked out so this test
    exercises only the wiring (call order/args/error-isolation), never real
    network/DB/file I/O."""
    mocks = {
        "macro": mocker.patch("src.psx_predictor.pipelines.orchestrator.run_macro_pipeline"),
        "flows": mocker.patch("src.psx_predictor.pipelines.orchestrator.run_institutional_flows_pipeline"),
        "fundamentals": mocker.patch("src.psx_predictor.pipelines.orchestrator.run_fundamentals_pipeline"),
        "news": mocker.patch("src.psx_predictor.pipelines.orchestrator.run_news_sentiment_pipeline"),
        "build_features": mocker.patch("src.psx_predictor.pipelines.orchestrator.build_features"),
    }
    # Phases 4.5/4.6 import their targets lazily inside the function body -
    # patch them at their own source module so that late import still
    # resolves to the mock.
    mocker.patch("src.psx_predictor.data.fetch_psx_market_stats.sync_psx_market_stats")
    mocker.patch("src.psx_predictor.scraper.psx_dps_market_scraper.PsxDpsMarketScraper")
    mocker.patch("src.psx_predictor.data.fetch_search_trends.sync_search_trends_to_db")
    mocker.patch("src.psx_predictor.data.fetch_sbp_additional.fetch_sbp_additional")
    mocker.patch("src.psx_predictor.data.fetch_pakistan_activity.fetch_pakistan_activity")
    mocker.patch("src.psx_predictor.data.feature_political_events.build_political_features")
    mocks["build_features"].return_value = pd.DataFrame({"a": [1]})
    return mocks


def test_runs_all_phases_in_order_for_given_tickers(mocker):
    mocks = _patch_all_subpipelines(mocker)

    result = run_full_data_pipeline(["ABL", "HBL"])

    assert result is True
    mocks["macro"].assert_called_once_with()
    mocks["flows"].assert_called_once_with(["ABL", "HBL"])
    mocks["fundamentals"].assert_called_once_with(["ABL", "HBL"])
    mocks["news"].assert_called_once_with(["ABL", "HBL"])
    assert mocks["build_features"].call_count == 2
    mocks["build_features"].assert_any_call("ABL")
    mocks["build_features"].assert_any_call("HBL")


def test_defaults_to_pso_mebl_when_no_tickers_given(mocker):
    mocks = _patch_all_subpipelines(mocker)

    run_full_data_pipeline(None)

    mocks["flows"].assert_called_once_with(["PSO", "MEBL"])
    mocks["fundamentals"].assert_called_once_with(["PSO", "MEBL"])


def test_one_tickers_build_features_failure_does_not_stop_others(mocker):
    """A per-ticker build_features error is caught and logged, not allowed to
    abort feature generation for the rest of the batch - critical for a
    104-ticker run where one bad ticker shouldn't kill the whole refresh."""
    mocks = _patch_all_subpipelines(mocker)
    mocks["build_features"].side_effect = [Exception("boom"), pd.DataFrame({"a": [1]})]

    result = run_full_data_pipeline(["BAD", "GOOD"])

    assert result is True
    assert mocks["build_features"].call_count == 2


def test_macro_pipeline_runs_once_regardless_of_ticker_count(mocker):
    """Macro/central-bank data is shared across all tickers, not per-ticker -
    it must run exactly once even for a large batch."""
    mocks = _patch_all_subpipelines(mocker)
    run_full_data_pipeline(["A", "B", "C", "D", "E"])
    mocks["macro"].assert_called_once()

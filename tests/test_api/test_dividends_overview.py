from collections import namedtuple
from datetime import date

from src.psx_predictor.api.routers.dividends import (
    _dividend_news,
    _estimate_next_payout,
    _meeting_date,
    _news_kind,
    _upcoming_meetings,
)

Notice = namedtuple("Notice", "symbol published title url")
TODAY = date(2026, 10, 16)


def test_estimate_follows_quarterly_calendar():
    # Uneven quarters: the next one mirrors last year's Mar payout, not last + 91 days.
    paid = [(date(2025, 3, 10), 5), (date(2025, 5, 2), 5), (date(2025, 8, 20), 5),
            (date(2025, 11, 5), 5), (date(2026, 3, 9), 5), (date(2026, 5, 4), 5),
            (date(2026, 8, 18), 5)]
    assert _estimate_next_payout(paid, 4, TODAY) == date(2026, 11, 5)


def test_estimate_annual_payer_a_year_on():
    paid = [(date(2024, 9, 20), 10), (date(2025, 9, 18), 12), (date(2026, 9, 17), 15)]
    assert _estimate_next_payout(paid, 1, TODAY) == date(2027, 9, 17)


def test_estimate_keeps_recently_due_payout():
    # Due yesterday but not yet on record: still shown, as due now.
    paid = [(date(2024, 10, 14), 10), (date(2025, 10, 15), 12)]
    assert _estimate_next_payout(paid, 1, TODAY) == date(2026, 10, 15)


def test_no_estimate_for_irregular_or_single_payout():
    assert _estimate_next_payout([(date(2025, 3, 1), 5), (date(2026, 3, 1), 5)], None, TODAY) is None
    assert _estimate_next_payout([(date(2026, 3, 1), 5)], 1, TODAY) is None


def test_meeting_date_formats():
    published = date(2026, 10, 5)
    assert _meeting_date("173 Board Meeting of FCCL - 27 October 2026", published) == date(2026, 10, 27)
    assert _meeting_date("PIBT BOARD MEETING - OCTOBER 21, 2026", published) == date(2026, 10, 21)
    assert _meeting_date("Board Meeting on 22-10-2026", published) == date(2026, 10, 22)


def test_period_end_dates_are_not_meeting_dates():
    published = date(2026, 10, 13)
    assert _meeting_date("Board Meeting for Results for the Quarter ended September 30, 2026", published) is None
    assert _meeting_date("BOARD MEETING 30-06-2026", date(2026, 9, 29)) is None


def test_upcoming_meetings_uses_latest_notice():
    notices = [
        Notice("FCCL", date(2026, 10, 5), "Board Meeting of FCCL - 27 October 2026", "u1"),
        Notice("HBL", date(2026, 10, 13), "Board Meeting for Financial Results", "u2"),
        Notice("KEL", date(2026, 10, 12), "Board Meeting Other Than Financial Results", "u3"),
        Notice("YOUW", date(2026, 10, 10), "Board Meeting", "u4"),
        Notice("YOUW", date(2026, 10, 11), "Board MeetingREVOKED", "u5"),
        Notice("EFERT", date(2026, 9, 20), "Board Meeting for Financial Results", "u6"),
        Notice("MARI", date(2026, 10, 2), "Notice of Board Meeting", "u7"),
        Notice("MARI", date(2026, 10, 5), "Results of Board Meeting", "u8"),
    ]
    meetings = _upcoming_meetings(notices, TODAY)
    assert meetings == {
        "FCCL": (date(2026, 10, 27), date(2026, 10, 5), "u1"),
        "HBL": (None, date(2026, 10, 13), "u2"),
    }


def test_news_kinds():
    assert _news_kind("Credit of Final Cash Dividend") == "paid"
    assert _news_kind("Lucky Cement credits final cash dividend of Rs. 5") == "paid"
    assert _news_kind("DECLARATION OF INTERIM DIVIDEND") == "declared"
    assert _news_kind("Hum Network Proposes Final Cash Dividend") == "declared"
    assert _news_kind("Notice of Interim Cash Dividend and Share Book Closure") == "book_closure"
    assert _news_kind("Dividend stocks to watch") == "news"


def test_dividend_news_dedupes_and_filters():
    names = {"LUCK": "Lucky Cement", "FFC": "Fauji Fertilizer"}
    notices = [
        ("LUCK", date(2026, 10, 9), "Credit of final cash dividend", "psx", None),
        ("LUCK", date(2026, 10, 1), "Unclaimed Dividend", "psx", None),
        ("XYZ", date(2026, 10, 9), "Credit of final cash dividend", "psx", None),
        ("FFC", date(2026, 10, 9), "Board Meeting", "psx", None),
    ]
    articles = [
        ("FFC", date(2026, 10, 12), "FFC Rs. 14.50 Dividend Declared - TechJuice", "a", "TechJuice"),
        ("FFC", date(2026, 10, 7), "FFC Rs. 14.50 Dividend Declared - TechJuice", "b", "TechJuice"),
    ]
    news = _dividend_news(notices, articles, names)
    assert [(n["symbol"], n["date"], n["kind"], n["source"], n["title"]) for n in news] == [
        ("FFC", "2026-10-12", "declared", "TechJuice", "FFC Rs. 14.50 Dividend Declared"),
        ("LUCK", "2026-10-09", "paid", "PSX", "Credit of final cash dividend"),
    ]

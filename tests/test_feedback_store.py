import sqlite3

import pytest

from brand_compliance.feedback_store import FeedbackEntry, FeedbackStore


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "feedback.db"))


def entry(is_correct=True, source_pdf="a.pdf"):
    return FeedbackEntry(
        ad_text="Buy now!",
        rule_text="No urgency language",
        rule_category="language",
        is_correct=is_correct,
        session_id="s1",
        source_pdf=source_pdf,
    )


def test_empty_store(store):
    assert store.count() == 0
    assert store.stats() == {"total": 0, "correct": 0, "incorrect": 0}


def test_add_and_get_all(store):
    store.add(entry(is_correct=False))
    assert store.get_all() == [("Buy now!", "No urgency language", "language", 0)]


def test_stats_counts_correct_and_incorrect(store):
    store.add(entry(True))
    store.add(entry(True))
    store.add(entry(False))
    assert store.stats() == {"total": 3, "correct": 2, "incorrect": 1}


def test_source_pdf_is_stored_but_get_all_returns_every_row(store):
    store.add(entry(source_pdf="a.pdf"))
    store.add(entry(source_pdf="b.pdf"))
    with sqlite3.connect(store.db_path) as conn:
        pdfs = conn.execute("SELECT source_pdf FROM feedback ORDER BY id").fetchall()
    assert pdfs == [("a.pdf",), ("b.pdf",)]
    assert len(store.get_all()) == 2


def test_count_since(store):
    store.add(entry())
    assert store.count_since("2000-01-01 00:00:00") == 1
    assert store.count_since("2999-01-01 00:00:00") == 0


def test_reopening_existing_database_keeps_rows(tmp_path):
    path = str(tmp_path / "feedback.db")
    FeedbackStore(db_path=path).add(entry())
    assert FeedbackStore(db_path=path).count() == 1


def test_count_since_compares_iso_timestamps_across_formats(store):
    from datetime import datetime, timedelta, timezone

    store.add(entry())
    now = datetime.now(timezone.utc)
    one_hour = timedelta(hours=1)

    assert store.count_since((now - one_hour).isoformat(timespec="seconds")) == 1
    assert store.count_since((now + one_hour).isoformat(timespec="seconds")) == 0
    # Naive "T" timestamps are read as UTC.
    assert store.count_since((now - one_hour).replace(tzinfo=None).isoformat(timespec="seconds")) == 1
    # The same instants written with a +05:00 offset.
    plus5 = timezone(timedelta(hours=5))
    assert store.count_since((now - one_hour).astimezone(plus5).isoformat(timespec="seconds")) == 1
    assert store.count_since((now + one_hour).astimezone(plus5).isoformat(timespec="seconds")) == 0

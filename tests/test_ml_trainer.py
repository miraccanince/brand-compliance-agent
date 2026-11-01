import json
import zlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from brand_compliance import ml_trainer
from brand_compliance.feedback_store import FeedbackEntry, FeedbackStore


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_trainer, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(ml_trainer, "HISTORY_PATH", str(tmp_path / "history.json"))
    return tmp_path


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic 8-dim vectors so training runs without Ollama."""
    def embed(texts):
        return np.array([np.random.default_rng(zlib.crc32(t.encode())).normal(size=8) for t in texts])
    monkeypatch.setattr(ml_trainer, "_embeddings", embed)


@pytest.fixture
def no_embeddings(monkeypatch):
    def fail(texts):
        raise AssertionError("embeddings must not be computed")
    monkeypatch.setattr(ml_trainer, "_embeddings", fail)


def make_store(tmp_path, correct, wrong):
    store = FeedbackStore(db_path=str(tmp_path / "feedback.db"))
    for i in range(correct + wrong):
        store.add(FeedbackEntry(
            ad_text=f"ad {i}", rule_text=f"rule {i}", rule_category="language",
            is_correct=i < correct, session_id="s1", source_pdf=f"pdf{i % 3}.pdf",
        ))
    return store


def test_predict_returns_none_without_trained_model(paths):
    assert ml_trainer.predict("some ad", "some rule") is None
    assert ml_trainer.model_exists() is False


def test_train_rejects_too_few_samples(paths, no_embeddings):
    store = make_store(paths, correct=5, wrong=3)
    with pytest.raises(ValueError, match="at least 10 samples"):
        ml_trainer.train(store)


@pytest.mark.parametrize("correct, wrong", [(10, 0), (0, 10), (9, 1), (1, 9)])
def test_train_rejects_missing_or_single_example_class(paths, no_embeddings, correct, wrong):
    store = make_store(paths, correct, wrong)
    with pytest.raises(ValueError, match="at least 2 examples of each label"):
        ml_trainer.train(store)
    assert not ml_trainer.model_exists()
    assert ml_trainer.get_history() == []


def test_train_reports_evaluation_size_and_baseline(paths, fake_embeddings):
    store = make_store(paths, correct=8, wrong=2)

    result = ml_trainer.train(store)

    assert result["samples"] == 10  # all feedback, whichever PDF it came from
    assert "scope" not in result
    assert result["test_samples"] == 2
    # With 8 correct and 2 wrong, the 2-row test split holds only "correct" labels,
    # so a constant guess scores 1.0. The baseline makes such a result visible.
    assert result["baseline_accuracy"] == 1.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert ml_trainer.model_exists()
    assert ml_trainer.get_history() == [result]


def test_trained_at_is_utc_with_offset(paths, fake_embeddings):
    result = ml_trainer.train(make_store(paths, correct=6, wrong=4))
    trained_at = datetime.fromisoformat(result["trained_at"])
    assert trained_at.utcoffset() == timedelta(0)
    assert abs(datetime.now(timezone.utc) - trained_at) < timedelta(minutes=1)


def test_samples_since_last_training(paths):
    store = make_store(paths, correct=3, wrong=2)
    assert ml_trainer.samples_since_last_training(store) == 5  # no history yet

    hour = timedelta(hours=1)
    trained_at = (datetime.now(timezone.utc) - hour).isoformat(timespec="seconds")
    with open(ml_trainer.HISTORY_PATH, "w") as f:
        json.dump([{"trained_at": trained_at}], f)
    assert ml_trainer.samples_since_last_training(store) == 5

    trained_at = (datetime.now(timezone.utc) + hour).isoformat(timespec="seconds")
    with open(ml_trainer.HISTORY_PATH, "w") as f:
        json.dump([{"trained_at": trained_at}], f)
    assert ml_trainer.samples_since_last_training(store) == 0

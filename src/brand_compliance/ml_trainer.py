import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from brand_compliance.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)

MODEL_PATH   = "custom_classifier.pkl"
HISTORY_PATH = "training_history.json"
MIN_SAMPLES  = 10
MIN_PER_CLASS = 2


@lru_cache(maxsize=1)
def _get_embed_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


def _embeddings(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    return np.array([model.embed_query(t) for t in texts])


def train(store: FeedbackStore) -> dict:
    """
    Train a logistic regression on all user feedback (one global model).

    Feature: concat(ad_text_embedding, rule_embedding)  →  768 + 768 = 1536 dims
    Label:   is_correct (1 = real violation, 0 = false positive)

    Raises ValueError before any embedding work if there are too few samples
    or fewer than MIN_PER_CLASS examples of either label (the stratified
    train/test split needs at least two of each).
    """
    rows = store.get_all()
    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} samples, have {len(rows)}")

    labels = np.array([r[3] for r in rows])
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if min(positives, negatives) < MIN_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_PER_CLASS} examples of each label, "
            f"have {positives} correct and {negatives} wrong. Add more varied feedback."
        )

    logger.info(f"Starting training | samples={len(rows)}")
    t0 = time.time()

    ad_texts   = [r[0] for r in rows]
    rule_texts = [r[1] for r in rows]

    logger.info("Computing embeddings for training data...")
    ad_embs   = _embeddings(ad_texts)
    rule_embs = _embeddings(rule_texts)
    X = np.concatenate([ad_embs, rule_embs], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    # Accuracy of always predicting the most common test label, for comparison.
    baseline = max(y_test.mean(), 1 - y_test.mean())
    logger.info(
        f"Training complete | accuracy={accuracy:.2f} on {len(y_test)} test samples "
        f"(majority baseline={baseline:.2f}) | elapsed={time.time()-t0:.1f}s"
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved | path={MODEL_PATH}")

    result = {
        "accuracy":          round(accuracy, 3),
        "baseline_accuracy": round(float(baseline), 3),
        "test_samples":      len(y_test),
        "samples":           len(rows),
        "trained_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_history(result)
    return result


def _append_history(entry: dict) -> None:
    history = get_history()
    history.append(entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_history() -> list[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)


def samples_since_last_training(store: FeedbackStore) -> int:
    history = get_history()
    if not history:
        return store.count()
    last_trained_at = history[-1]["trained_at"]
    return store.count_since(last_trained_at)


def predict(ad_text: str, rule_text: str) -> float | None:
    """
    Predict P(real violation) using the trained custom model.
    Returns None if no model has been trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    model = _get_embed_model()
    ad_emb   = np.array(model.embed_query(ad_text))
    rule_emb = np.array(model.embed_query(rule_text))
    x = np.concatenate([ad_emb, rule_emb]).reshape(1, -1)

    prob = clf.predict_proba(x)[0][1]
    logger.debug(f"Custom model prediction | prob={prob:.2f} | rule='{rule_text[:50]}'")
    return float(prob)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

import json
import logging
import os
import pickle
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from feedback_store import FeedbackStore

logger = logging.getLogger(__name__)

MODEL_PATH   = "custom_classifier.pkl"
HISTORY_PATH = "training_history.json"
MIN_SAMPLES  = 10


@lru_cache(maxsize=1)
def _get_embed_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


def _embeddings(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    return np.array([model.embed_query(t) for t in texts])


def train(store: FeedbackStore, source_pdf: str | None = None) -> dict:
    """
    Train a logistic regression on user feedback.

    Feature: concat(ad_text_embedding, rule_embedding)  →  768 + 768 = 1536 dims
    Label:   is_correct (1 = real violation, 0 = false positive)
    """
    rows = store.get_all(source_pdf=source_pdf)
    scope = f"pdf={source_pdf}" if source_pdf else "all PDFs"
    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} samples for {scope}, have {len(rows)}")

    logger.info(f"Starting training | scope={scope} | samples={len(rows)}")
    t0 = time.time()

    ad_texts   = [r[0] for r in rows]
    rule_texts = [r[1] for r in rows]
    labels     = np.array([r[3] for r in rows])

    logger.info("Computing embeddings for training data...")
    ad_embs   = _embeddings(ad_texts)
    rule_embs = _embeddings(rule_texts)
    X = np.concatenate([ad_embs, rule_embs], axis=1)

    if len(set(labels)) < 2:
        raise ValueError("Need both positive and negative examples to train. Add more varied feedback.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    logger.info(f"Training complete | accuracy={accuracy:.2f} | elapsed={time.time()-t0:.1f}s")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved | path={MODEL_PATH}")

    result = {
        "accuracy":   round(accuracy, 3),
        "samples":    len(rows),
        "scope":      source_pdf or "all",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
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

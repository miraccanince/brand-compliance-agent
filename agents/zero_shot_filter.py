import logging
import time
from functools import lru_cache

from models import BrandRule

logger = logging.getLogger(__name__)

_MODEL     = "facebook/bart-large-mnli"
_THRESHOLD = 0.50
_MAX_CHARS = 512  # BART tokenizer limit


@lru_cache(maxsize=1)
def _load_classifier():
    from transformers import pipeline
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Loading zero-shot classifier | model={_MODEL} | device={device}")
    logger.info("First run: model will be downloaded (~1.6GB), this takes a few minutes...")
    t0 = time.time()
    clf = pipeline("zero-shot-classification", model=_MODEL, device=device)
    logger.info(f"Classifier ready | elapsed={time.time()-t0:.1f}s")
    return clf


def filter_rules(
    ad_text: str,
    rules: list[BrandRule],
    threshold: float = _THRESHOLD,
) -> tuple[dict[str, float], list[BrandRule]]:
    """
    Zero-shot NLI pre-filter.
    Returns: (scores_per_rule, rules_likely_violated)

    Uses BART-MNLI to estimate P(ad violates rule) for each rule.
    Rules above threshold are passed to the LLM for detailed analysis.
    """
    if not rules:
        return {}, []

    t0 = time.time()
    clf = _load_classifier()

    # Use rule text as labels — multi_label=True means each rule gets
    # an independent sigmoid score (not softmax), suitable for multi-violation
    result = clf(
        ad_text[:_MAX_CHARS],
        candidate_labels=[r.rule for r in rules],
        hypothesis_template="This advertisement violates the brand guideline: {}",
        multi_label=True,
    )

    score_map: dict[str, float] = dict(zip(result["labels"], result["scores"]))

    filtered: list[BrandRule] = []
    for rule in rules:
        score = score_map.get(rule.rule, 0.0)
        if score >= threshold:
            filtered.append(rule)
            logger.warning(f"ML violation detected | score={score:.2f} | rule='{rule.rule[:60]}'")
        else:
            logger.debug(f"ML passed | score={score:.2f} | rule='{rule.rule[:60]}'")

    logger.info(
        f"Zero-shot filter | rules={len(rules)} → likely_violated={len(filtered)} "
        f"| threshold={threshold} | elapsed={time.time()-t0:.1f}s"
    )
    return score_map, filtered

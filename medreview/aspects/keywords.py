# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Keyword-matching aspect classifier (the baseline)."""
import re
from .lexicon import SIDE_EFFECT_CUES, EFFECTIVENESS_CUES
from .mentions import symptom_mentions

EFFECTIVENESS, SIDE_EFFECTS, NEITHER = "effectiveness", "side_effects", "neither"
LABELS = [EFFECTIVENESS, SIDE_EFFECTS, NEITHER]


def _compile(patterns):
    return [re.compile(r"\b(?:" + p + r")\b", re.I) for p in patterns]


def _any(patterns):
    # One combined pattern used as a fast pre-check before counting individual cues.
    return re.compile(r"\b(?:" + "|".join("(?:" + p + ")" for p in patterns) + r")\b", re.I)


def _count(sentence, combined, individual):
    return sum(1 for r in individual if r.search(sentence)) if combined.search(sentence) else 0


class KeywordAspectClassifier:
    """Labels a sentence by counting keyword hits for each aspect.

    A specific symptom term (e.g. "nausea") counts double, since it is stronger
    evidence than a generic cue, unless it names the problem being treated
    ("helps my nausea"), in which case it counts toward effectiveness instead.
    Ties go to side effects; no hits means "neither".

    `condition` (the review's condition field) is optional; when given, symptom
    terms that match it are treated as the condition, not a side effect.
    """

    def __init__(self, symptom_weight: float = 2.0):
        self.symptom_weight = symptom_weight
        self._eff = _compile(EFFECTIVENESS_CUES)
        self._se_cues = _compile(SIDE_EFFECT_CUES)
        self._eff_any, self._se_any = _any(EFFECTIVENESS_CUES), _any(SIDE_EFFECT_CUES)

    def scores(self, sentence: str, condition: str | None = None) -> dict[str, float]:
        eff = _count(sentence, self._eff_any, self._eff)
        se = _count(sentence, self._se_any, self._se_cues)
        for m in symptom_mentions(sentence, condition):
            if m.treated:
                eff += 0.5
            else:
                se += self.symptom_weight
        return {EFFECTIVENESS: float(eff), SIDE_EFFECTS: float(se)}

    def predict_one(self, sentence: str, condition: str | None = None) -> str:
        s = self.scores(sentence, condition)
        if s[EFFECTIVENESS] == 0 and s[SIDE_EFFECTS] == 0:
            return NEITHER
        return SIDE_EFFECTS if s[SIDE_EFFECTS] >= s[EFFECTIVENESS] else EFFECTIVENESS

    def predict(self, sentences, conditions=None) -> list[str]:
        """conditions: None, one string for all sentences, or one per sentence."""
        sentences = list(sentences)
        if conditions is None or isinstance(conditions, str):
            conditions = [conditions] * len(sentences)
        return [self.predict_one(s, c if isinstance(c, str) else None) for s, c in zip(sentences, conditions)]

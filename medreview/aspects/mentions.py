# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Find side-effect terms in a sentence and decide how each is being used."""
import re
from typing import NamedTuple

from .lexicon import SIDE_EFFECT_TERMS, NEGATIONS, TREATING_VERBS, CONDITION_HINTS

_TERM_RES = {name: re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.I)
             for name, pats in SIDE_EFFECT_TERMS.items()}
# A negation word within the three words before the term: "no nausea", "didn't have any cramps".
_NEG = re.compile(r"(?:\b(?:" + "|".join(NEGATIONS) + r")\b|n't)(?:\W+\w+){0,3}\W*$", re.I)
_FILLER = r"(?:\W+(?:my|the|his|her|their|our|severe|chronic|bad|major|terrible|horrible|constant|daily|of|with|and|or))*"
# "helps my chronic anxiety", "for depression", "my migraines": the term is the
# problem being treated, not something the drug caused.
_TREATING = re.compile(
    r"\b(?:" + "|".join(TREATING_VERBS) + r")(?:\W+\w+){0,2}" + _FILLER + r"\W*$"
    r"|\b(?:for|my|his|her|with)" + _FILLER + r"\W*$", re.I)
# ...unless the drug is clearly the cause: "gave me my first migraine".
_CAUSED = re.compile(r"\b(?:gave|gives|giving|made|makes|making|caus\w+|side[- ]?effects?)\b[^.;]*$", re.I)
_ANY_TERM = re.compile(r"\b(?:" + "|".join(p for pats in SIDE_EFFECT_TERMS.values() for p in pats) + r")\b", re.I)
_LIST_SEP = re.compile(r"(?:,|\band\b|\bor\b|&|/)\s*$", re.I)


def treats(name: str, condition: str | None) -> bool:
    """True if side effect `name` is what `condition` is (e.g. "headache" for "Migraine")."""
    cond = condition.lower() if isinstance(condition, str) else ""
    return bool(cond) and (name in cond or any(h in cond for h in CONDITION_HINTS.get(name, ())))


class Mention(NamedTuple):
    name: str        # canonical side effect, e.g. "nausea"
    text: str        # matched words
    negated: bool    # "no nausea"
    treated: bool    # the condition being treated, e.g. "helps my nausea"


def symptom_mentions(sentence: str, condition: str | None = None) -> list[Mention]:
    """First mention of each lexicon term in the sentence, with its context flags."""
    if not _ANY_TERM.search(sentence):
        return []
    cond = condition.lower() if isinstance(condition, str) else ""
    out, spans = [], []
    for name, rx in _TERM_RES.items():
        m = rx.search(sentence)
        if m:
            spans.append((m.start(), name, m))
    prev = None
    for start, name, m in sorted(spans, key=lambda t: t[0]):
        before = sentence[:start]
        text = m.group(0).lower()
        negated = bool(_NEG.search(before))
        treated = treats(name, cond) or (bool(cond) and text in cond)
        if not treated and _TREATING.search(before) and not _CAUSED.search(before):
            treated = True
        # Items in a list share the first item's context: "helps my anxiety, insomnia and nausea".
        if prev is not None and _LIST_SEP.search(before) and before.rstrip(" ,&/andor").endswith(prev.text):
            negated, treated = negated or prev.negated, treated or prev.treated
        prev = Mention(name, text, negated, treated)
        out.append(prev)
    return out

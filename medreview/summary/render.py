# AI-assisted code
# Model: Opus 5.5 (max effort)
"""HTML pieces for the Streamlit app. All review text is escaped before it is marked up.

Colors (checked with a color-vision-deficiency validator, light and dark themes):
  sentences about effectiveness  aqua wash   rgba(27,175,122,.26)
  sentences about side effects   amber wash  rgba(237,161,0,.68)
  effectiveness sentiment        blue #2a78d6 / gray #a8a79f / red #e34948
"""
import html
import re

from ..aspects import EFFECTIVENESS, SIDE_EFFECTS
from ..aspects.lexicon import SIDE_EFFECT_TERMS
from .analysis import SENTIMENTS

SENTIMENT_NAMES = {"positive": "Positive", "neutral": "Mixed or neutral", "negative": "Negative"}
ASPECT_CLASSES = {EFFECTIVENESS: "mr-eff", SIDE_EFFECTS: "mr-se"}
ASPECT_NAMES = {EFFECTIVENESS: "About effectiveness", SIDE_EFFECTS: "About side effects"}

CSS = """
<style>
.mr-eff, .mr-se { border-radius: 3px; padding: 0.05em 0.15em; -webkit-box-decoration-break: clone;
  box-decoration-break: clone; }
.mr-eff { background: rgba(27, 175, 122, 0.26); }
.mr-se { background: rgba(237, 161, 0, 0.68); }
.mr-review { line-height: 1.65; max-width: 40rem; margin: 0.25rem 0 0.5rem; }
.mr-review b, .mr-quote b { font-weight: 700; text-decoration: underline; text-decoration-thickness: 1px;
  text-underline-offset: 2px; }
.mr-meta { font-size: 0.85rem; opacity: 0.75; margin: 0 0 0.2rem; }
.mr-seg { display: flex; gap: 2px; height: 28px; margin: 0.4rem 0 0.45rem; max-width: 40rem; }
.mr-part { border-radius: 4px; display: flex; align-items: center; min-width: 3px; }
.mr-part span { padding: 0 8px; font-size: 0.8rem; font-weight: 600; white-space: nowrap; }
.mr-positive { background: #2a78d6; color: #ffffff; }
.mr-neutral { background: #a8a79f; color: #0b0b0b; }
.mr-negative { background: #e34948; color: #ffffff; }
.mr-legend { display: flex; flex-wrap: wrap; gap: 0.35rem 1.1rem; font-size: 0.82rem; opacity: 0.85;
  margin: 0 0 0.3rem; }
.mr-key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px;
  vertical-align: -1px; }
.mr-figure { font-size: 2.6rem; font-weight: 650; line-height: 1.05; margin: 0.15rem 0 0.1rem; }
.mr-figure-label { font-size: 0.95rem; margin: 0 0 0.35rem; }
.mr-quote { border-left: 3px solid rgba(127, 127, 127, 0.35); padding: 0.1rem 0 0.1rem 0.7rem;
  margin: 0.35rem 0; font-size: 0.92rem; max-width: 40rem; }
.mr-quote .mr-meta { font-size: 0.78rem; }
</style>
"""


def term_pattern(keywords=(), side_effects=()):
    """A regex for the search words (whole words) and the lexicon variants of the side
    effects asked about, or None when there is nothing to mark."""
    parts = [re.escape(k) for k in keywords if k]
    for name in side_effects:
        parts.extend(SIDE_EFFECT_TERMS.get(name, ()))
    if not parts:
        return None
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def mark(text: str, pattern=None) -> str:
    """Escape `text` and wrap matches of `pattern` in <b>."""
    if pattern is None:
        return html.escape(text)
    out, last = [], 0
    for match in pattern.finditer(text):
        if match.end() == match.start():
            continue
        out.append(html.escape(text[last:match.start()]))
        out.append("<b>" + html.escape(match.group(0)) + "</b>")
        last = match.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def review_html(review, pattern=None) -> str:
    """The review's sentences, highlighted by aspect, with search words in bold."""
    if not review.sentences:
        return f'<div class="mr-review">{mark(review.review, pattern)}</div>'
    parts = []
    for sentence, aspect in review.sentences:
        body = mark(sentence, pattern)
        css = ASPECT_CLASSES.get(aspect)
        parts.append(f'<span class="{css}" title="{ASPECT_NAMES[aspect]}">{body}</span>' if css else body)
    return '<div class="mr-review">' + " ".join(parts) + "</div>"


def aspect_legend() -> str:
    return ('<div class="mr-legend"><span class="mr-eff">About effectiveness</span>'
            '<span class="mr-se">About side effects</span><span>Not highlighted: other sentences</span>'
            '<span><b>Bold</b>: your search words</span></div>')


def sentiment_bar(counts, label: str) -> str:
    """A segmented bar of effectiveness sentiment. Shares print inside blocks wide enough."""
    total = counts.total
    if not total:
        return ""
    blocks, spoken = [], []
    for key in SENTIMENTS:
        n = getattr(counts, key)
        share = n / total
        spoken.append(f"{share:.0%} {SENTIMENT_NAMES[key].lower()}")
        if not n:
            continue
        text = f"<span>{share:.0%}</span>" if share >= 0.08 else ""
        blocks.append(f'<div class="mr-part mr-{key}" style="flex: {n} 1 0" '
                      f'title="{SENTIMENT_NAMES[key]}: {n} of {total} reviews ({share:.0%})">{text}</div>')
    aria = html.escape(f"{label}: " + ", ".join(spoken))
    return f'<div class="mr-seg" role="img" aria-label="{aria}">' + "".join(blocks) + "</div>"


def sentiment_legend() -> str:
    keys = "".join(f'<span><i class="mr-key mr-{k}"></i>{SENTIMENT_NAMES[k]}</span>' for k in SENTIMENTS)
    return f'<div class="mr-legend">{keys}</div>'


def quote_html(review, sentence: str, pattern=None) -> str:
    """One sentence from a review, with where it came from."""
    meta = f"Rated {review.rating}/10" + (f" for {html.escape(review.condition)}" if review.condition else "")
    if review.useful_count:
        meta += f". {review.useful_count:,} found this review useful"
    return f'<div class="mr-quote">{mark(sentence, pattern)}<p class="mr-meta">{meta}.</p></div>'

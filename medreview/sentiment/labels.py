"""Turn user ratings into sentiment labels to score classifiers against."""
NEGATIVE, NEUTRAL, POSITIVE = "negative", "neutral", "positive"
LABELS = [NEGATIVE, NEUTRAL, POSITIVE]


def rating_to_label(rating) -> str:
    """1-4 negative, 5-6 neutral, 7-10 positive."""
    rating = int(rating)
    if not 1 <= rating <= 10:
        raise ValueError(f"rating must be 1-10, got {rating}")
    if rating <= 4:
        return NEGATIVE
    if rating <= 6:
        return NEUTRAL
    return POSITIVE

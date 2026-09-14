CONFIDENCE_THRESHOLD = 0.75

def filter_labels(labels: list[dict]) -> tuple[list, list]:
    """Split into (auto-accepted, needs-review)."""
    accepted, review = [], []
    for lbl in labels:
        if lbl.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
            accepted.append(lbl)
        else:
            review.append(lbl)
    return accepted, review
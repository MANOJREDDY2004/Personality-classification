import re


def clean(s: str) -> str:
    """Lowercase, strip URLs and @mentions, collapse whitespace."""
    s = re.sub(r"@\w+|http\S+|www\S+", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

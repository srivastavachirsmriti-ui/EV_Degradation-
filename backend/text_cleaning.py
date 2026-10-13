"""Text cleaning reproduced from the training pipeline.

The training data ships with a ``clean_text`` column. Comparing every raw
``post_text`` against its ``clean_text`` counterpart showed the exact transform
below reproduces the stored values for 100% of the 2,500 rows:

1. lowercase
2. replace every character that is not ``a-z``, ``0-9``, ``.``, ``%`` or
   whitespace with a single space
3. collapse runs of whitespace and strip the ends

Periods and percent signs are deliberately preserved (e.g. ``98.8%``,
``vw id.3``); commas, hyphens, question marks, semicolons, etc. are dropped.
"""

from __future__ import annotations

import re

_NON_KEEP = re.compile(r"[^a-z0-9.%\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Return the cleaned representation used during model training."""
    lowered = str(text).lower()
    spaced = _NON_KEEP.sub(" ", lowered)
    return _WHITESPACE.sub(" ", spaced).strip()

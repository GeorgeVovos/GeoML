"""v06 wrapper around shared/model_classical_logreg.

v01-v06 did not include an L1-regularised linear baseline, which is the
historical small-data champion on gene-expression data. This file plugs
the gap without modifying earlier results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.model_classical_logreg import train_classical_logreg  # noqa: E402,F401

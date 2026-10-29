"""Single place for input paths used by the paper-2 operators (operators/<operator>.py).

Every paper-2 operator reads data ONLY through this module, only from datasets/
(paths relative to the repository root), offline, $0. No API calls. Per-area loaders live in operators/p2parts/<area>.py
and are re-exported here, so operators write `from p2io import ...` or `import p2io; p2io.<area>.<loader>`.
The raw runs (GPQA texts, raw run logs) are not published and are never read by operators.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUT2 = ROOT / "summaries" / "out2"
EXPECTED2 = ROOT / "tests" / "expected2"     # frozen outputs of the original scripts (regression targets)
sys.path.insert(0, str(Path(__file__).resolve().parent))

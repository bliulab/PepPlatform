"""Compatibility command for frozen data-preparation recipes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pepplatform.task_compat import LEGACY_HLAI_COLUMN, LEGACY_HLAI_ROOT
from tools.decode_hlai_pseudoseq import main as _decode_main


def main(argv=None):
    return _decode_main(
        argv, defaults_root="datasets/" + LEGACY_HLAI_ROOT, output_column=LEGACY_HLAI_COLUMN
    )


if __name__ == "__main__":
    main()

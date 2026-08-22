"""Keep the deterministic unit-test suite independent of downloaded ML assets."""

import os


os.environ.setdefault("BANKREG_BGE_MODE", "disabled")

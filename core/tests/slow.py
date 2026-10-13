"""Marker for long tests (> 2 s): they run only in the full suite, `REFGRADE_SLOW=1`.

The fast suite (default, used by the pipeline after every change) must stay under 60 s.
Slow tests are not weaker, just run less often: once per package and before a release.
"""

import os
import unittest

FULL = os.environ.get("REFGRADE_SLOW") == "1"
slow = unittest.skipUnless(FULL, "langsam: volle Suite mit REFGRADE_SLOW=1")

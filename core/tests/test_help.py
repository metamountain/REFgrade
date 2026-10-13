"""Tests for refgrade/help.py: every slider has exactly one complete HELP entry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.help import HELP
from refgrade.sliders import SLIDERS


class TestHelp(unittest.TestCase):

    def test_all_sliders_have_help(self):
        missing = {s.name for s in SLIDERS} - set(HELP)
        self.assertEqual(missing, set(), f"HELP missing for SLIDERS: {sorted(missing)}")

    def test_no_orphan_entries(self):
        orphans = set(HELP) - {s.name for s in SLIDERS}
        self.assertEqual(orphans, set(), f"Orphan HELP entries: {sorted(orphans)}")

    def test_entries_have_en_and_tip(self):
        for name, entry in HELP.items():
            self.assertTrue(entry.get("en", "").strip(), f"HELP[{name!r}].en missing/empty")
            self.assertTrue(entry.get("tip", "").strip(), f"HELP[{name!r}].tip missing/empty")


if __name__ == "__main__":
    unittest.main()

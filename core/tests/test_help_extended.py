"""Tests for the extended help texts (K29g).

Checks:
- Every SECTION_HELP entry has the three headings and is ≥ 600 chars.
- Every VISIBLE slider has HELP[name]["en"] ≥ 60 chars and a non-empty ["tip"].
- HelpPopup offscreen 2560×1392: height ≤ 0.7 × dialog height, scroll area, bold, monospace.
- HTML escaping: '<' in the color section is escaped to &lt; in the popup.
- Sample math: wheels text contains "0.15" and "p = 0.5 - 0.35"; color text contains "20 * temp".
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

# Set up the ref_grade package without triggering the krita import in __init__.py
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from refgrade.help import HELP  # noqa: E402
from refgrade.sections import VISIBLE  # noqa: E402
from ref_grade.ui_help import SECTION_HELP  # noqa: E402


class TestSectionHelp(unittest.TestCase):
    """Every SECTION_HELP entry must have the K33a headings and be ≥ 600 chars."""

    def test_all_entries_have_headings(self) -> None:
        self.assertGreater(len(SECTION_HELP), 0, "SECTION_HELP is empty")
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                self.assertIn("In short", text, f"Missing 'In short' in {key!r}")
                self.assertIn("Background", text, f"Missing 'Background' in {key!r}")
                self.assertIn("Technical", text, f"Missing 'Technical' in {key!r}")

    def test_all_entries_min_length(self) -> None:
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                self.assertGreaterEqual(
                    len(text), 600,
                    f"SECTION_HELP[{key!r}] is {len(text)} chars, need ≥ 600"
                )

    def test_expected_keys_present(self) -> None:
        expected = {"reference", "tone", "color", "hsl", "curves", "wheels", "lut", "rgb", "general"}
        missing = expected - set(SECTION_HELP.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


class TestSliderHelp(unittest.TestCase):
    """Every VISIBLE slider must have HELP[name]['en'] ≥ 60 chars and a non-empty ['tip']."""

    def test_all_visible_sliders(self) -> None:
        for name in VISIBLE:
            with self.subTest(slider=name):
                self.assertIn(name, HELP, f"Slider {name!r} not in HELP")
                entry = HELP[name]
                self.assertIn("en", entry, f"HELP[{name!r}] missing 'en'")
                self.assertIn("tip", entry, f"HELP[{name!r}] missing 'tip'")
                self.assertGreaterEqual(
                    len(entry["en"]), 60,
                    f"HELP[{name!r}]['en'] is {len(entry['en'])} chars, need ≥ 60"
                )
                self.assertTrue(
                    entry["tip"].strip(),
                    f"HELP[{name!r}]['tip'] is empty"
                )


class TestHelpPopup(unittest.TestCase):
    """HelpPopup offscreen: height ≤ 0.7 × dialog height, scroll area, bold, monospace, escaping."""

    def test_popup_tone_offscreen(self) -> None:
        from PyQt6.QtWidgets import QApplication, QScrollArea, QWidget

        app = QApplication.instance() or QApplication(sys.argv)

        parent = QWidget()
        parent.resize(2560, 1392)
        parent.show()

        from ref_grade.dialog import HelpPopup
        popup = HelpPopup("tone", 800, parent)
        popup.show()

        # Height must be ≤ 70% of parent height
        max_expected = int(1392 * 0.7)
        self.assertLessEqual(
            popup.height(), max_expected,
            f"Popup height {popup.height()} exceeds 70% of dialog ({max_expected})"
        )

        # Must contain a QScrollArea
        scroll = popup.findChild(QScrollArea)
        self.assertIsNotNone(scroll, "HelpPopup does not contain a QScrollArea")

        # Headings must be rendered bold
        html = popup._label.text()
        self.assertIn("font-weight:bold", html, "Headings should be bold in the HTML")

        # Formulas must be rendered in monospace
        self.assertIn("monospace", html, "Formulas should be in monospace in the HTML")

        popup.close()
        parent.close()

    def test_popup_color_escapes_html(self) -> None:
        """The color section contains '<' which must be escaped in the HTML output."""
        from PyQt6.QtWidgets import QApplication, QWidget

        app = QApplication.instance() or QApplication(sys.argv)

        parent = QWidget()
        parent.resize(2560, 1392)
        parent.show()

        from ref_grade.dialog import HelpPopup
        popup = HelpPopup("color", 800, parent)
        popup.show()

        html = popup._label.text()
        # The raw '<' from "20 < L* < 95" must be escaped to &lt;
        self.assertIn("&lt;", html, "'<' in the color help must be escaped to &lt;")
        # The unescaped '<' followed by 'L*' must NOT appear (would be a broken tag)
        self.assertNotIn("< L*", html, "Unescaped '<' before 'L*' would break the HTML")

        popup.close()
        parent.close()


class TestMathContent(unittest.TestCase):
    """Sample math checks against the actual code constants."""

    def test_wheels_contains_lift_range(self) -> None:
        self.assertIn("0.15", SECTION_HELP["wheels"],
                      "Wheels help should mention the 0.15 lift range at the rim")

    def test_wheels_contains_balance_formula(self) -> None:
        self.assertIn("p = 0.5 - 0.35", SECTION_HELP["wheels"],
                      "Wheels help should contain the balance formula p = 0.5 - 0.35*Balance")

    def test_color_contains_temp_factor(self) -> None:
        self.assertIn("2^(0.8·temp)", SECTION_HELP["color"],
                      "Color help should give the von Kries gains of _temp_tint (2026-10-13)")


if __name__ == "__main__":
    unittest.main()

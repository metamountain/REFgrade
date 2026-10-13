"""Tests for the K33a help structure (In short → Background → Technical → Controls).

Checks:
- Every original SECTION_HELP key still exists.
- Each section has headings in order: In short → Background → Technical → Controls.
- In short has 4–7 bullet points; Background ≥ 150 words.
- controls_table(key) contains every slider from SECTIONS[key] exactly once,
  with range from SLIDERS (fixed: exposure "-4.00 … 4.00", default "0.00").
- Every VISIBLE slider has HELP[n]["short"] with ≤ 8 words.
- No "Pedestal" in SECTION_HELP or any short/tip/en text.
"""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
if "ref_grade" not in sys.modules:
    _pkg = types.ModuleType("ref_grade")
    _pkg.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["ref_grade"] = _pkg

os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from refgrade.help import HELP  # noqa: E402
from refgrade.sections import SECTIONS, VISIBLE  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402
from ref_grade.ui_help import SECTION_HELP, controls_table, _LABELS  # noqa: E402

# Keys that must exist (from the original SECTION_HELP).
_REQUIRED_KEYS = frozenset({
    "general", "reference", "tone", "color", "wheels", "rgb", "hsl", "curves", "lut",
})

_HEADING_ORDER = ("In short", "Background", "Technical", "Controls")


def _word_count(text: str) -> int:
    return len(text.split())


def _in_short_bullets(text: str) -> list[str]:
    """Extract bullet lines between **In short** and the next ** heading."""
    m = re.search(r"\*\*In short\*\*\n(.*?)(?=\n\*\*|\Z)", text, re.DOTALL)
    if not m:
        return []
    return [line for line in m.group(1).splitlines() if line.startswith("- ")]


def _background_text(text: str) -> str:
    """Extract the Background section text (between **Background** and next ** heading)."""
    m = re.search(r"\*\*Background\*\*\n(.*?)(?=\n\*\*|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _heading_positions(text: str) -> list[int]:
    """Return the character position of each required heading, in order."""
    positions = []
    for heading in _HEADING_ORDER:
        idx = text.find(f"**{heading}**")
        positions.append(idx)
    return positions


class TestSectionHelpKeys(unittest.TestCase):
    """Every original SECTION_HELP key still exists."""

    def test_all_keys_present(self) -> None:
        missing = _REQUIRED_KEYS - set(SECTION_HELP.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


class TestSectionHelpStructure(unittest.TestCase):
    """Each section has headings in order; In short 4–7 bullets; Background ≥ 150 words."""

    def test_headings_in_order(self) -> None:
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                # Controls is generated at runtime by controls_table(), so check combined text
                combined = text + "\n" + controls_table(key)
                positions = _heading_positions(combined)
                self.assertNotIn(-1, positions,
                                 f"{key!r}: missing one of {_HEADING_ORDER}")
                self.assertEqual(positions, sorted(positions),
                                 f"{key!r}: headings not in order {positions}")

    def test_in_short_bullet_count(self) -> None:
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                bullets = _in_short_bullets(text)
                self.assertGreaterEqual(len(bullets), 4,
                                        f"{key!r}: In short has {len(bullets)} bullets, need ≥ 4")
                self.assertLessEqual(len(bullets), 7,
                                     f"{key!r}: In short has {len(bullets)} bullets, need ≤ 7")

    def test_background_min_words(self) -> None:
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                bg = _background_text(text)
                wc = _word_count(bg)
                self.assertGreaterEqual(wc, 150,
                                        f"{key!r}: Background has {wc} words, need ≥ 150")


class TestControlsTable(unittest.TestCase):
    """controls_table(key) contains every slider from SECTIONS[key] exactly once."""

    def test_all_sliders_present(self) -> None:
        for key, names in SECTIONS.items():
            with self.subTest(section=key):
                table = controls_table(key)
                for name in names:
                    label = _LABELS.get(name, name)
                    self.assertIn(label, table,
                                  f"{key!r}: label {label!r} (for {name!r}) not in controls_table")

    def test_exposure_range_fixed(self) -> None:
        table = controls_table("tone")
        self.assertIn("-4.00 … 4.00", table,
                      f"exposure range not '-4.00 … 4.00' in controls_table('tone')")
        # Default 0.00 for exposure
        self.assertIn("0.00", table)

    def test_slider_count_matches(self) -> None:
        for key, names in SECTIONS.items():
            with self.subTest(section=key):
                table = controls_table(key)
                if not names:
                    self.assertIn("no sliders", table)
                    continue
                lines = [l for l in table.splitlines() if l.startswith("- ")]
                self.assertEqual(len(lines), len(names),
                                 f"{key!r}: {len(lines)} lines, expected {len(names)}")


class TestShortField(unittest.TestCase):
    """Every VISIBLE slider has HELP[n]['short'] with ≤ 8 words."""

    def test_all_visible_have_short(self) -> None:
        for name in VISIBLE:
            with self.subTest(slider=name):
                h = HELP.get(name)
                self.assertIsNotNone(h, f"HELP missing entry for {name!r}")
                self.assertIn("short", h, f"HELP[{name!r}] missing 'short' field")
                wc = _word_count(h["short"])
                self.assertLessEqual(wc, 8,
                                     f"HELP[{name!r}]['short'] has {wc} words, need ≤ 8: {h['short']!r}")


class TestNoPedestal(unittest.TestCase):
    """No 'Pedestal' in SECTION_HELP or any short/tip/en text."""

    def test_no_pedestal_in_section_help(self) -> None:
        for key, text in SECTION_HELP.items():
            with self.subTest(section=key):
                self.assertNotIn("Pedestal", text,
                                 f"SECTION_HELP[{key!r}] contains 'Pedestal'")
                self.assertNotIn("pedestal", text,
                                 f"SECTION_HELP[{key!r}] contains 'pedestal'")

    def test_no_pedestal_in_help_texts(self) -> None:
        for name, h in HELP.items():
            for field in ("short", "tip", "en"):
                with self.subTest(slider=name, field=field):
                    text = h.get(field, "")
                    self.assertNotIn("Pedestal", text,
                                     f"HELP[{name!r}]['{field}'] contains 'Pedestal'")
                    self.assertNotIn("pedestal", text,
                                     f"HELP[{name!r}]['{field}'] contains 'pedestal'")


class TestNoDrift(unittest.TestCase):
    """The plugin-side copies must equal their core / dialog originals."""

    def test_sections_copy_matches_core(self) -> None:
        from ref_grade import ui_help
        self.assertEqual(ui_help._SECTIONS, SECTIONS)

    def test_labels_copy_matches_dialog_labels(self) -> None:
        from ref_grade import dlg_common
        for name, label in dlg_common.LABELS.items():
            with self.subTest(slider=name):
                self.assertEqual(ui_help_labels().get(name), label)

    def test_rgb_offset_is_shown_as_offset(self) -> None:
        rows = [l for l in controls_table("rgb").splitlines() if l.startswith("- M Offset ")]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].endswith("| -0.50 … 0.50 | 0.00"), rows[0])


class TestSliderTooltip(unittest.TestCase):
    def test_short_is_bold_and_tip_below(self) -> None:
        from ref_grade.ui_help import slider_tooltip
        self.assertEqual(slider_tooltip({"short": "Brightness", "tip": "Use <small>"}),
                         "<b>Brightness</b><br>Use &lt;small&gt;")

    def test_no_help_gives_empty_tooltip(self) -> None:
        from ref_grade.ui_help import slider_tooltip
        self.assertEqual(slider_tooltip({}), "")


def ui_help_labels() -> dict:
    from ref_grade import ui_help
    return ui_help._LABELS


if __name__ == "__main__":
    unittest.main()

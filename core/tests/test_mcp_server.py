"""K32a: Color-Grading-MCP-Server — Scope-Zahlen, Werkzeuge über den echten Worker, MCP-Protokoll."""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

import mcp_server as M  # noqa: E402
from refgrade.scope_stats import stats  # noqa: E402
from refgrade.sections import SECTIONS  # noqa: E402

POOL = CORE.parent / "bench" / "pool"


class TestScopeStats(unittest.TestCase):
    def test_grey_neutral(self):
        s = stats(torch.full((1, 32, 32, 3), 0.5))
        for zone in ("mids", "all"):
            self.assertLessEqual(abs(s["cast"][zone]["a"]), 0.5)
            self.assertLessEqual(abs(s["cast"][zone]["b"]), 0.5)
        self.assertTrue(all(v == 0 for v in s["clip"].values()))
        self.assertAlmostEqual(s["channels"]["luma"]["p50"], 0.5, delta=0.01)
        self.assertAlmostEqual(sum(s["hist16"]), 1.0, delta=1e-3)

    def test_white_clips_and_red_cast(self):
        self.assertEqual(stats(torch.ones(1, 8, 8, 3))["clip"]["r_hi"], 1.0)
        red = torch.zeros(1, 8, 8, 3)
        red[..., 0] = 1.0
        self.assertGreater(stats(red)["cast"]["all"]["a"], 40)


class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = M.open_image(str(POOL / "04.jpg"))
        cls.sid = cls.info["session"]

    @classmethod
    def tearDownClass(cls):
        M.shutdown()

    def test_open_size(self):
        w, h = Image.open(POOL / "04.jpg").size
        self.assertEqual((self.info["width"], self.info["height"]), (w, h))

    def test_preview_jpeg_and_clamp(self):
        data, text = M.preview(self.sid, {"exposure": 99})
        img = Image.open(io.BytesIO(data))
        self.assertEqual(img.format, "JPEG")
        self.assertLessEqual(max(img.size), 768)
        self.assertIn("4.0", text)
        self.assertIn("exposure", text)

    def test_unknown_slider_lists_valid(self):
        with self.assertRaises(M.GradeError) as cm:
            M.preview(self.sid, {"gibtsnicht": 1})
        self.assertIn("exposure", str(cm.exception))

    def test_scopes_follow_exposure(self):
        p0 = M.scopes(self.sid, {})["channels"]["luma"]["p50"]
        p1 = M.scopes(self.sid, {"exposure": 1.0})["channels"]["luma"]["p50"]
        self.assertGreaterEqual(p1, p0 + 0.05)

    def test_auto(self):
        vals = M.auto(self.sid, "tone")["values"]
        self.assertTrue(set(vals) <= set(SECTIONS["tone"]))
        with self.assertRaises(M.GradeError):
            M.auto(self.sid, "lut")

    def test_exports(self):
        d = tempfile.mkdtemp(prefix="mcp_exp_")
        cube = M.export_lut(self.sid, {"exposure": 0.3}, d)
        self.assertIn("LUT_3D_SIZE", Path(cube).read_text(encoding="utf-8", errors="replace"))
        out = M.export_image(self.sid, {"exposure": 0.3}, os.path.join(d, "o.png"))
        self.assertEqual(Image.open(out).size, (self.info["width"], self.info["height"]))


class TestProtocol(unittest.TestCase):
    def test_list_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def run() -> list[str]:
            params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"],
                                           cwd=str(CORE))
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return sorted(t.name for t in (await s.list_tools()).tools)

        self.assertEqual(asyncio.run(run()), sorted(M.TOOL_NAMES))


if __name__ == "__main__":
    unittest.main()

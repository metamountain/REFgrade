"""Tooltip texts for non-slider widgets, keyed by objectName.

One sentence per widget. Slider tooltips are built separately from HELP + SLIDERS.
SECTION_HELP holds the full help text for the "?" popup per section.
controls_table(key) generates the Controls table at runtime from SECTIONS, SLIDERS, HELP, LABELS.
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path

# ── Section → slider mapping (mirrors core/refgrade/sections.py SECTIONS) ────
_SECTIONS: dict[str, tuple[str, ...]] = {
    "reference": ("light_strength", "color_strength", "protect_shadows", "protect_highlights"),
    "tone": ("exposure", "contrast", "highlights", "highlight_rolloff", "highlight_reconstruct",
             "shadows", "whites", "blacks", "clarity", "texture", "dehaze"),
    "color": ("temperature", "tint", "saturation", "vibrance"),
    "wheels": tuple(f"{k}_{c}" for k in ("lift", "gamma", "gain") for c in ("m", "r", "g", "b"))
             + ("wheel_blending", "wheel_balance", "wheel_intensity"),
    "rgb": tuple(f"rgb_{p}_{k}" for p in ("m", "r", "g", "b") for k in ("gamma", "pedestal", "gain"))
           + tuple(f"rgb_use_{z}" for z in ("shadows", "midtones", "highlights")),
    "hsl": tuple(f"sel_{b}_{p}" for b in ("red", "yellow", "green", "cyan", "blue", "magenta")
                 for p in ("hue", "sat", "light")),
    "color_replace": tuple(f"cr_{i}_{k}" for i in range(8)
                           for k in ("src_hue", "src_sat", "src_bright", "tgt_hue", "tgt_sat", "tgt_bright", "intensity", "range", "sat_floor")),
    "curves": (),
    "lut": (),
}

# ── Display names (mirrors dlg_common.LABELS for the controls table) ─────────
_LABELS: dict[str, str] = {
    "light_strength": "Light", "color_strength": "Color",
    "protect_shadows": "Protect shadows", "protect_highlights": "Protect highlights",
    "exposure": "Exposure", "contrast": "Contrast",
    "highlights": "Highlights", "highlight_rolloff": "Recover",
    "highlight_reconstruct": "Reconstruct",
    "shadows": "Shadows",
    "whites": "Whites", "blacks": "Blacks",
    "clarity": "Clarity", "texture": "Texture", "dehaze": "Dehaze",
    "temperature": "Temperature", "tint": "Tint",
    "saturation": "Saturation", "vibrance": "Vibrance",
    "wheel_blending": "Blending", "wheel_balance": "Balance",
    "wheel_intensity": "Intensity",
    "lift_m": "M Lift", "lift_r": "R Lift", "lift_g": "G Lift", "lift_b": "B Lift",
    "gamma_m": "M Gamma", "gamma_r": "R Gamma", "gamma_g": "G Gamma", "gamma_b": "B Gamma",
    "gain_m": "M Gain", "gain_r": "R Gain", "gain_g": "G Gain", "gain_b": "B Gain",
    "rgb_m_gamma": "M Gamma", "rgb_m_pedestal": "M Offset", "rgb_m_gain": "M Gain",
    "rgb_r_gamma": "R Gamma", "rgb_r_pedestal": "R Offset", "rgb_r_gain": "R Gain",
    "rgb_g_gamma": "G Gamma", "rgb_g_pedestal": "G Offset", "rgb_g_gain": "G Gain",
    "rgb_b_gamma": "B Gamma", "rgb_b_pedestal": "B Offset", "rgb_b_gain": "B Gain",
    "rgb_use_shadows": "Shadows", "rgb_use_midtones": "Mids", "rgb_use_highlights": "Highs",
    "sel_red_hue": "Red Hue", "sel_red_sat": "Red Sat", "sel_red_light": "Red Light",
    "sel_yellow_hue": "Yellow Hue", "sel_yellow_sat": "Yellow Sat", "sel_yellow_light": "Yellow Light",
    "sel_green_hue": "Green Hue", "sel_green_sat": "Green Sat", "sel_green_light": "Green Light",
    "sel_cyan_hue": "Cyan Hue", "sel_cyan_sat": "Cyan Sat", "sel_cyan_light": "Cyan Light",
    "sel_blue_hue": "Blue Hue", "sel_blue_sat": "Blue Sat", "sel_blue_light": "Blue Light",
    "sel_magenta_hue": "Magenta Hue", "sel_magenta_sat": "Magenta Sat", "sel_magenta_light": "Magenta Light",
    "shadow_hue": "Shadow Hue", "shadow_amount": "Shadow Amount",
    "highlight_hue": "Highlight Hue", "highlight_amount": "Highlight Amount",
    **{f"cr_{i}_{k}": f"CR{i+1} {k.replace('_', ' ').title()}"
       for i in range(8) for k in ("src_hue", "src_sat", "src_bright", "tgt_hue", "tgt_sat", "tgt_bright", "intensity", "range", "sat_floor")},
}


def _load_sliders() -> list:
    """Load SLIDERS from core by file path (no torch needed)."""
    if "ref_grade_sliders" in sys.modules:
        return sys.modules["ref_grade_sliders"].SLIDERS
    core_dir = Path(__file__).resolve().parents[2] / "core"
    spec = importlib.util.spec_from_file_location("ref_grade_sliders",
                                                  core_dir / "refgrade" / "sliders.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_grade_sliders"] = mod
    spec.loader.exec_module(mod)
    return mod.SLIDERS


def _load_help() -> dict:
    """Load HELP from core by file path (no torch needed)."""
    if "ref_grade_help" in sys.modules:
        return sys.modules["ref_grade_help"].HELP
    core_dir = Path(__file__).resolve().parents[2] / "core"
    spec = importlib.util.spec_from_file_location("ref_grade_help",
                                                  core_dir / "refgrade" / "help.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_grade_help"] = mod
    spec.loader.exec_module(mod)
    return mod.HELP


def slider_tooltip(h: dict) -> str:
    """Rich-text tooltip for a slider: ``short`` in bold, ``tip`` below; empty if no HELP entry."""
    if not h:
        return ""
    return f"<b>{html.escape(h.get('short', ''))}</b><br>{html.escape(h.get('tip', ''))}"


def _fmt(v: float) -> str:
    """Format a number for display: 2 decimal places."""
    return f"{v:.2f}"


def controls_table(key: str) -> str:
    """Generate the Controls section text for a section key.

    Returns a Markdown-like table: **Controls** header followed by one line per slider:
    ``- Name — short | range | default``.
    """
    sliders = _load_sliders()
    help_data = _load_help()
    by_name = {s.name: s for s in sliders}
    names = _SECTIONS.get(key, ())
    if not names:
        return "**Controls**\n- (no sliders in this section)"
    lines = ["**Controls**"]
    for name in names:
        s = by_name.get(name)
        h = help_data.get(name, {})
        label = _LABELS.get(name, name)
        short = h.get("short", "")
        if s:
            rng = f"{_fmt(s.min)} … {_fmt(s.max)}"
            default = _fmt(s.default)
        else:
            rng = "—"
            default = "—"
        lines.append(f"- {label} — {short} | {rng} | {default}")
    return "\n".join(lines)


SECTION_HELP: dict[str, str] = {
    "general": (
        "**In short**\n"
        "- Transfers light and color mood from a reference image\n"
        "- Pick a reference, adjust with FIT or manually\n"
        "- Preview on the right, OK applies as a new layer\n"
        "- Chapters run as a chain, each on the result of the previous; MASTER scales the whole chain\n"
        "- Chain order: Reference → Color → Tone → RGB → Curves → Wheels → HSL → LUT (⇅ changes it)\n"
        "- Use FIT for a good base, then fine-tune\n"
        "**Background**\n"
        "RefGrade is a color-grading tool that works like a digital darkroom. "
        "You start with a reference image — a photo whose light and color you like — "
        "and the tool analyses how that image differs from yours in terms of brightness, "
        "contrast, and color temperature. It then applies a similar transformation to your image. "
        "Think of it like holding up a colored glass in front of a picture to change its mood, "
        "but with precise, adjustable controls.\n"
        "The workflow is simple: (1) Load a reference image into one of the tiles at the top. "
        "(2) Press FIT to get an automatic starting point. (3) Adjust the sliders in each section "
        "to fine-tune the look — Temperature for warm/cool, Contrast for punch, Wheels for "
        "creative tinting. (4) Watch the scopes at the bottom to check for clipping or color "
        "balance. (5) Press OK to apply the result as a new layer in Krita.\n"
        "Common mistakes: setting all sections to 100 % at once (the effects stack and can "
        "oversaturate); ignoring the scopes and clipping highlights; using a reference with a "
        "very different subject (a landscape reference on a portrait often looks wrong). "
        "In the scopes: the HIST shows your tonal distribution — if the neon lines hit the top "
        "or bottom edge, you are clipping. The Vectorscope shows color balance — if the cloud "
        "is shifted to one side, the image has a color cast.\n"
        "Sections interact as a chain in the order Reference → Color → Tone → RGB → Curves → "
        "Wheels → HSL → LUT: each chapter works on the result of the previous one. Reference "
        "sets the base mood and shifts the starting point for everything after it. Tone changes "
        "brightness. Wheels add creative tint. HSL and Curves give fine per-color control. The "
        "RGB section is for precise video-style correction.\n"
        "**Technical**\n"
        "Chain: `y_i = y_{i-1} + w_i * (f_i(y_{i-1}) - y_{i-1})`, then "
        "`R(x) = clamp(x + master * (y_n - x), 0, 1)`, where f_i is each chapter's "
        "sRGB-to-sRGB function, w_i its Amount (0-1), and n the last chapter. "
        "# sections.py:chain_fn\n"
        "Each section's function is built independently and combined; the result is baked "
        "into a `33^3` LUT for fast preview. # session.py:build_lut\n"
        "Spatial effects (Clarity, Texture, Dehaze) are applied after the LUT at full "
        "resolution. # session.py:preview\n"
        "Processing chain in `grade.grade`: exposure → temp/tint → LGG (wheels) → contrast → "
        "tone zones → saturation → vibrance → selective color (HSL) → split toning. "
        "Chapters run as a chain in `session.build_lut` (Reference → Color → Tone → RGB → "
        "Curves → Wheels → HSL → LUT).\n"
        "Color space: sRGB (0..1 float32). Lab (D65) for white balance and reference transfer. "
        "HSV for selective color. Rec.709 luma for zone masks.\n"
        "Code: `core/refgrade/grade.py` (manual chain), `core/refgrade/sections.py` (chain), "
        "`core/refgrade/session.py` (LUT build + preview).\n"
        "**Practical — real scenarios**\n"
        "- **Starting a new grade** → Problem: you have a photo and want to grade it. "
        "Solution: (1) Load a reference image into a tile. (2) Press FIT for an automatic "
        "starting point. (3) Adjust the sections from top to bottom (Reference → Color → "
        "Tone → RGB → Curves → Wheels → HSL → LUT). (4) Check the scopes. (5) Press OK.\n"
        "- **All sections at 100 % looks oversaturated** → Problem: you turned on all "
        "sections at full strength and the image looks like a poster. Solution: keep most "
        "sections at a moderate strength (50–80 %). The effects stack as a chain, so each "
        "section works on the result of the previous one. Use the Amount slider per section "
        "to control the strength.\n"
        "- **Clipping in the scopes** → Problem: the HIST neon lines hit the top or bottom "
        "edge, or the PARADE peaks hit the top. Solution: reduce the section that is causing "
        "the clipping (usually Exposure, Contrast, or Whites). Check the PARADE to see which "
        "channel is clipping.\n"
        "- **Color cast (image looks tinted)** → Problem: the image has an orange or blue "
        "cast. Solution: go to the Color section and press AUTO WB, then fine-tune Temperature "
        "and Tint. Check the Vectorscope — the cloud should be roughly centered.\n"
        "- **Chain order matters** → Problem: the result looks wrong, and you are not sure "
        "why. Solution: the sections run as a chain in the order Reference → Color → Tone → "
        "RGB → Curves → Wheels → HSL → LUT. Each section works on the result of the previous "
        "one. Use the ⇅ button to change the order if you want a different workflow.\n"
        "- **Preview is slow** → Problem: the preview takes a long time to update. Solution: "
        "the preview is a 33³ LUT, which is fast. If it is slow, the image is very large or "
        "the spatial effects (Clarity, Texture, Dehaze) are active (they run at full "
        "resolution). Temporarily turn off the spatial effects to speed up the preview.\n"
    ),
    "reference": (
        "**In short**\n"
        "- Transfers light and color mood from the active reference tile\n"
        "- Method: Reinhard (soft), MKL (strong), Zonal (per-zone), PAIR (before/after)\n"
        "- Light and Color strength control how much transfers\n"
        "- Protect sliders keep specific tonal zones closer to original\n"
        "- After FIT, Reference is set to 0 (sliders already captured the look)\n"
        "**Background**\n"
        "The Reference section is the heart of RefGrade. It takes a reference image — a photo "
        "whose atmosphere you want to copy — and transfers its light and color characteristics "
        "to your image. Think of it like a photographer saying \"I want my portrait to feel "
        "like that golden-hour landscape photo.\" The tool analyses the reference's brightness "
        "distribution and color palette, then applies a similar transformation to your image.\n"
        "There are four methods. Reinhard is the gentlest: it matches the mean and standard "
        "deviation of brightness and color in a perceptual color space (Lab). MKL is stronger: "
        "it also matches the full color covariance, so it captures color relationships more "
        "faithfully. Zonal splits the image into 17 brightness zones and transfers each one "
        "separately, giving more control over shadows vs. highlights. PAIR is different: instead "
        "of a reference image, you give it a before/after pair (two photos of the same scene, "
        "one graded) and it learns the transformation directly.\n"
        "To use it: (1) Drag or load a reference image into a tile. (2) Choose a method — start "
        "with Reinhard. (3) Press FIT to auto-set the sliders. (4) Adjust Light and Color "
        "strength (0.5 is a good start). (5) Use Protect if shadows or highlights look wrong.\n"
        "Common mistakes: using a reference with very different exposure (raise Light to 0.7-0.8); "
        "not protecting shadows (dark areas pick up unwanted reference brightness); using PAIR "
        "with misaligned images (the before and after must show the exact same scene).\n"
        "In the scopes: after applying a reference, check the HIST to see if the tonal "
        "distribution shifted toward the reference. The Vectorscope shows if the color palette "
        "moved toward the reference's colors. If the Vectorscope cloud becomes very small, the "
        "image lost color — reduce Color strength.\n"
        "Interplay: Reference sets the base mood. Color (white balance) should be applied after "
        "to fine-tune the temperature. Wheels can add creative tint on top. HSL can protect "
        "specific colors (e.g., keep skin tones natural while the rest shifts).\n"
        "**Practical — real scenarios**\n"
        "- **Portrait should feel like a golden-hour landscape** → Problem: you have a flat "
        "studio portrait and a reference photo with warm, dramatic light. Solution: load the "
        "reference into a tile, pick Reinhard (the gentlest method), press FIT, then set Light "
        "to 0.6 and Color to 0.5. The portrait now carries the reference's warm mood without "
        "losing the subject's detail.\n"
        "- **Reference exposure is very different** → Problem: the reference is much brighter "
        "or darker than your image, so the transfer looks wrong. Solution: raise Light to "
        "0.7–0.8 to compensate, or use Zonal (it transfers per brightness zone and handles "
        "exposure gaps better than Reinhard).\n"
        "- **Shadows pick up unwanted reference brightness** → Problem: after the transfer, "
        "your dark areas are too bright or too colorful. Solution: raise Protect shadows to "
        "0.4–0.6 to keep the dark areas close to the original.\n"
        "- **Colors look oversaturated after transfer** → Problem: the reference's color "
        "palette is stronger than yours, so the result looks posterized. Solution: lower Color "
        "to 0.3–0.5, or use MKL (it matches the full color covariance and is more faithful).\n"
        "- **Before/after pair (same scene, one graded)** → Problem: you have two photos of "
        "the exact same scene, one already graded, and you want to learn the grade. Solution: "
        "use the PAIR method — first tile = before, second = after (or one side-by-side image). "
        "The tool learns the transformation directly and applies it.\n"
        "- **After FIT, Reference is set to 0** → This is intentional: the sliders already "
        "captured the look, so the Reference section is switched off to avoid double-applying. "
        "Turn it back on only if you want to blend more reference mood on top.\n"
        "**Technical**\n"
        "Reinhard: per Lab channel, `y = (x - mu_s) / sigma_s * sigma_r + mu_r`, blended as "
        "`out = x + s * (y - x)`. # transfer.py:reinhard\n"
        "MKL: `y = A * (x - mu_s) + mu_r`, where A is the 3x3 matrix that maps the source Lab "
        "covariance to the reference covariance. # transfer.py:mkl\n"
        "Zonal: 17-quantile L-curve + per-zone delta-a/delta-b offset + chroma scale "
        "(`sigma_ref/sigma_src`, clamped 0.25-4). # zonal.py:apply_params\n"
        "Protect: `out = orig + (matched - orig) * (1 - ps*w_sh - pm*w_mid - ph*w_hi)`, "
        "zone weights from Lab L*. # protect.py:protect\n"
        "Color space: CIELAB (D65). Reference and image are compared via statistics only "
        "(different sizes allowed). Subsampling ≤ 1 000 000 pixels, fixed seed.\n"
        "Code: `core/refgrade/transfer.py` (Reinhard, MKL), `core/refgrade/zonal.py` (Zonal), "
        "`core/refgrade/protect.py` (Protect), `core/refgrade/session.py` (PAIR via LUT fit).\n"
    ),
    "tone": (
        "**In short**\n"
        "- Exposure, contrast, and Lightroom-style zone adjustments\n"
        "- Highlights/Shadows/Whites/Blacks for zone-specific brightness\n"
        "- Clarity, Texture, Dehaze are spatial effects (not in LUT)\n"
        "- POP button applies a one-click recipe across all of these\n"
        "- Check the Parade scope for clipping after adjusting Whites/Blacks\n"
        "**Background**\n"
        "The Tone section controls the brightness and contrast of your image. Think of it like "
        "adjusting the lights in a room: Exposure is like opening or closing the window "
        "(more or less light overall). Contrast is like the difference between the brightest "
        "and darkest parts — high contrast makes a dramatic image, low contrast makes it soft "
        "and dreamy. The zone adjustments (Highlights, Shadows, Whites, Blacks) let you "
        "adjust specific brightness ranges independently, just like Lightroom.\n"
        "Highlights recovers detail in the brightest areas (like a blown-out sky). Shadows "
        "lifts the darkest areas (like revealing detail in a dark corner). Whites pushes the "
        "brightest pixels toward pure white. Blacks pushes the darkest pixels toward pure black. "
        "Clarity adds a subtle glow of micro-contrast in the midtones — it makes the image look "
        "more \"punchy\" without increasing overall contrast. Texture sharpens fine details. "
        "Dehaze removes the milky look caused by atmospheric particles.\n"
        "To use it: (1) Set Exposure so the image is correctly bright (check the HIST — the "
        "distribution should fill the range). (2) Set Contrast for the mood you want. (3) Use "
        "Highlights/Shadows to balance specific zones. (4) Add a touch of Clarity for punch. "
        "(5) Check the Parade scope — if a channel hits the top edge, reduce Gain or Whites.\n"
        "Common mistakes: too much Exposure (clipping in the HIST); too much Contrast (losing "
        "shadow detail); Clarity too high (halos around edges); Dehaze on an image that is not "
        "actually hazy (it adds contrast in a weird way).\n"
        "In the scopes: HIST shows the overall brightness distribution — after Exposure, the "
        "neon lines should span most of the range. PARADE shows R/G/B separately — peaks at the "
        "top edge mean clipping. WAVE shows brightness across the image width — dense white "
        "areas are highlights.\n"
        "Interplay: Tone works after Color (white balance) in the processing chain. If you "
        "change Exposure, you may need to re-adjust Contrast. Clarity, Texture, and Dehaze are "
        "spatial effects applied after the LUT, so they are not included in SAVE LUT.\n"
        "**Practical — real scenarios**\n"
        "- **Flat, lifeless image** → Problem: the photo looks washed out, no "
        "depth. Solution: raise Contrast to 1.1–1.2, add Clarity 0.2–0.3, and lift Blacks "
        "slightly (-0.1) to keep the shadows from crushing. Check the HIST — the distribution "
        "should spread across the range without hitting the edges.\n"
        "- **Blown-out sky (white, no detail)** → Problem: the sky is pure white, clouds "
        "invisible. Solution: drag Highlights to -0.3 to -0.5 to bring back the cloud detail. "
        "If the sky was already pure white in the original, no setting can restore it — use "
        "Highlight rolloff (0.3–0.5) to soften the transition.\n"
        "- **Crushed shadows (black, no detail)** → Problem: dark areas are pure black, "
        "you lose texture in clothing or foliage. Solution: drag Shadows to +0.2 to +0.4 to "
        "lift them. Watch the HIST — the left edge should not pile up at zero.\n"
        "- **Hazy, milky landscape** → Problem: distant mountains look faded, the image "
        "looks like it is seen through fog. Solution: raise Dehaze to 0.2–0.4. Do not use it "
        "on a portrait or a studio shot — it adds an unnatural contrast.\n"
        "- **Skin looks waxy or over-sharpened** → Problem: Clarity or Texture is too high, "
        "creating halos around the face. Solution: lower Clarity to 0.1 or 0, and Texture to "
        "0.1. For portraits, keep both below 0.2.\n"
        "- **One-click base** → Press POP to get a balanced starting point (Contrast 1.1, "
        "Vibrance 0.3, Highlights -0.35, Shadows 0.35, Whites 0.2, Blacks -0.2, Clarity 0.35). "
        "Then fine-tune from there.\n"
        "**Technical**\n"
        "Exposure: `x_lin = x_lin * 2^exposure` in linear RGB (sRGB decoded, multiplied, "
        "re-encoded). # grade.py:_exposure\n"
        "Contrast: `y = (x - pivot) * contrast + pivot`, then highlight rolloff: knee "
        "`k = 1 - 0.6*r`, luma `Y <= k` untouched, above `k` scaled by `Yn/Y` with "
        "`Yn = k + (1-k)*tanh((Y-k)/(1-k))` (hue preserved). r = 0 leaves the image bit-identical. "
        "# grade.py:_contrast\n"
        "Zones: `L' = L + 0.25 * (sh * bump(0.25) + hi * bump(0.75) + bl * (1-L)^3 + wh * L^3)`, "
        "enforced monotone via cumulative max on 1024 points. # grade.py:_tone_zones\n"
        "Clarity: Gaussian unsharp (sigma scaled to image size) on L* only, masked to "
        "midtones; `amount = 0.6 * clarity`. # clarity.py:apply\n"
        "Texture: high-pass (Gaussian, sigma ~ 2 px at 1024) on L*; `amount = 1.5 * texture`. "
        "# texture.py:apply\n"
        "Dehaze: Dark Channel Prior (He et al. 2009), `omega = 0.95 * |dehaze|, t_min = 0.1`. "
        "# dehaze.py:apply\n"
        "Processing chain position: exposure → temp/tint → LGG → contrast → tone zones → "
        "saturation → vibrance → HSL → split. Clarity/Texture/Dehaze: after LUT, spatial.\n"
        "Color space: sRGB (Exposure in linear RGB). L* (CIELAB) for zones. Code: "
        "`core/refgrade/grade.py`, `core/refgrade/clarity.py`, `core/refgrade/texture.py`, "
        "`core/refgrade/dehaze.py`.\n"
    ),
    "color": (
        "**In short**\n"
        "- White balance: Temperature (warm-cool) and Tint (green-magenta)\n"
        "- Saturation scales all colors; Vibrance boosts only pale colors\n"
        "- Use AUTO WB first to kill the cast, then fine-tune\n"
        "- PICK WB: click something that should be neutral grey\n"
        "- Temperature ±0.1-0.3 for subtle; ±0.5-1.0 for creative\n"
        "**Background**\n"
        "White balance is the process of making white look white. Cameras estimate the color "
        "of the light automatically, but they often get it wrong — a photo taken under tungsten "
        "light looks orange, one taken in open shade looks blue. White balance corrects this "
        "cast so that neutral colors (white, grey, black) appear neutral.\n"
        "Temperature shifts the image along the blue-yellow axis. Positive values make the "
        "image warmer (more yellow/orange), negative values make it cooler (more blue). Think "
        "of it like the time of day: morning light is cool (blue), sunset is warm (orange). "
        "Tint shifts along the green-magenta axis — it corrects the remaining cast after "
        "temperature is set. Positive = magenta (warms skin), negative = green (cools, good "
        "for outdoor).\n"
        "Saturation controls the intensity of all colors. Below 1.0, colors become muted and "
        "filmic. Above 1.0, they become vivid and punchy. Vibrance is smarter: it only boosts "
        "colors that are already pale, leaving saturated colors alone. This means you can add "
        "color to a dull image without making reds explode.\n"
        "To use it: (1) Press AUTO WB to neutralise the cast automatically. (2) Fine-tune "
        "Temperature for the mood (warm for cozy, cool for clinical). (3) Adjust Tint if skin "
        "tones look green or magenta. (4) Set Saturation for the overall color intensity. "
        "(5) Add Vibrance if the image still looks dull.\n"
        "Common mistakes: over-correcting white balance (the image looks unnatural); using "
        "Saturation > 1.3 (colors clip and look posterized); ignoring Vibrance (it is the "
        "safest way to add color).\n"
        "In the scopes: the Vectorscope shows color balance — if the cloud is shifted to one "
        "side, there is a cast. After AUTO WB, the cloud should be roughly centered. The HIST "
        "white ghost shows brightness — white balance should not change it significantly.\n"
        "Interplay: Color is the second chapter in the chain (after Reference, before Tone). "
        "Changing Temperature shifts the starting point for everything after it (Tone, RGB, "
        "Curves, Wheels, HSL). If you set a warm Temperature, you may need to reduce the warm "
        "tint in the Highlights wheel.\n"
        "**Practical — real scenarios**\n"
        "- **Photo looks orange (tungsten light)** → Problem: indoor photo under warm bulbs "
        "has a strong orange cast. Solution: press AUTO WB to neutralise it, then fine-tune "
        "Temperature to -0.2 to -0.4 if a slight warmth remains. Check the Vectorscope — the "
        "cloud should move toward the center.\n"
        "- **Photo looks blue (open shade)** → Problem: outdoor photo in shade looks cold and "
        "blue. Solution: AUTO WB, then Temperature +0.2 to +0.4 for warmth. If skin looks "
        "green, add Tint +0.1 to +0.2 (magenta).\n"
        "- **Skin tones look green or magenta** → Problem: after white balance, the face has "
        "a green or magenta cast. Solution: adjust Tint — positive (magenta) removes green, "
        "negative (green) removes magenta. Small steps of 0.05–0.1.\n"
        "- **Image looks dull and flat** → Problem: colors are muted, the photo looks faded. "
        "Solution: raise Saturation to 1.1–1.2, or better, use Vibrance 0.2–0.4 (it only "
        "boosts pale colors and spares the vivid ones, so reds do not explode).\n"
        "- **Colors look oversaturated / posterized** → Problem: Saturation is too high, the "
        "image looks like a poster. Solution: lower Saturation below 1.0 (0.8–0.9 for a "
        "filmic, muted look), or use Vibrance instead (it is safer).\n"
        "- **Neutral grey looks colored** → Problem: a grey wall or paper looks tinted. "
        "Solution: use the PICK WB eyedropper — click on the grey area, and Temperature and "
        "Tint follow to make it neutral.\n"
        "**Technical**\n"
        "Temperature/Tint: white balance (von Kries) in linear light, channel gains "
        "`R·2^(0.8·temp), G·2^(-0.5·tint), B·2^(-0.8·temp)`, normalised so grey keeps its "
        "brightness — a cast scales with brightness, so it is removed everywhere, not just in "
        "the mids. # grade.py:_temp_tint\n"
        "Saturation: `out = Y + s * (x - Y)`, where `Y = 0.2126*R + 0.7152*G + 0.0722*B`. "
        "# grade.py:_saturation\n"
        "Vibrance: `factor = 1 + v * (1 - sat)`, `out = Y + factor * (x - Y)`, where "
        "`sat = (max - min) / max`. # grade.py:_vibrance\n"
        "Auto WB: median linear colour (r, g, b) of near-neutral pixels (`20 < L* < 95, "
        "chroma < 25`); `temperature = log2(b/r)/1.6`, `tint = -2·(0.8·temperature + "
        "log2(r/g))` — the exact inverse of the gains above. # whitebalance.py:estimate\n"
        "Processing chain position: exposure → temp/tint → LGG → contrast → zones → "
        "saturation → vibrance → HSL → split. Color space: CIELAB (D65) for WB, sRGB for "
        "saturation/vibrance. Code: `core/refgrade/grade.py`, `core/refgrade/whitebalance.py`.\n"
    ),
    "hsl": (
        "**In short**\n"
        "- Per-color-band adjustments for six hue bands\n"
        "- Each band: shift hue, saturation, or luminance independently\n"
        "- Grey and near-grey pixels are protected\n"
        "- Use small values (±0.1-0.3) for subtle shifts\n"
        "- Red band: hue -10 to -20 moves skin from orange toward true red\n"
        "**Background**\n"
        "The HSL section lets you adjust specific colors in your image without affecting the "
        "others. Think of it like a painter who can change only the blue in the sky without "
        "touching the blue in the shirt. The image is divided into six color bands — Red, "
        "Yellow, Green, Cyan, Blue, Magenta — and each band can be shifted in hue (color "
        "direction), saturation (intensity), or luminance (brightness) independently.\n"
        "This is incredibly useful for portraits: if skin tones look too orange, shift the Red "
        "and Yellow bands slightly toward red (negative hue). If the sky is too saturated, "
        "reduce the Blue band's saturation. If the vegetation looks dull, increase the Green "
        "band's saturation. The key is to use small values — a shift of 10-20 degrees is "
        "usually enough for a natural look.\n"
        "The bands overlap smoothly: a pixel at 30° (orange) is affected by both the Red band "
        "(centered at 0°) and the Yellow band (centered at 60°), with weights that add up to 1. "
        "This means there are no hard boundaries between bands — the transitions are smooth.\n"
        "To use it: (1) Identify which color band you want to adjust. (2) Choose the mode: "
        "Hue (shift color direction), Saturation (more/less intense), or Lightness (lighter/"
        "darker). (3) Drag the slider — small values first. (4) Check the result in the image "
        "and the Vectorscope (the cloud should shift in the direction you expect).\n"
        "Common mistakes: too large a hue shift (colors become unnatural); adjusting a band "
        "that does not contain the color you want (check the Vectorscope); forgetting that "
        "grey pixels are protected (a desaturated red is not affected by the Red band).\n"
        "In the scopes: the Vectorscope is your best friend here — it shows the color "
        "distribution as a cloud. When you shift the Red band, the red part of the cloud moves. "
        "The HIST shows if luminance changes affect the overall brightness.\n"
        "Interplay: HSL is the seventh chapter in the chain (after Wheels, before LUT), so it "
        "fine-tunes the color after the global adjustments. Use it to protect specific colors "
        "(e.g., keep skin natural while the rest of the image shifts with Wheels).\n"
        "**Practical — real scenarios**\n"
        "- **Skin looks too orange** → Problem: the face has an orange cast that the global "
        "white balance did not fix. Solution: shift the Red band's Hue to -10 to -20 (moves "
        "skin from orange toward true red). Check the Vectorscope — the red part of the cloud "
        "should move toward magenta.\n"
        "- **Sky is too saturated** → Problem: the blue sky is vivid and draws the eye away "
        "from the subject. Solution: lower the Blue band's Saturation to -0.2 to -0.3 for a "
        "muted, filmic sky. The rest of the image keeps its color.\n"
        "- **Vegetation looks dull** → Problem: the green foliage looks faded. Solution: raise "
        "the Green band's Saturation to 0.1–0.3, and shift its Hue to +10 to +20 (toward cyan) "
        "for more natural vegetation.\n"
        "- **Wheels shifted the skin, but you want to protect it** → Problem: the Wheels added "
        "a creative tint, but the face got tinted too. Solution: use HSL to pull the Red and "
        "Yellow bands back — reduce their Saturation slightly and shift their Hue to compensate. "
        "HSL acts after Wheels, so it can undo the skin tint.\n"
        "- **Grey areas got tinted** → Note: grey and near-grey pixels are protected by a "
        "saturation guard, so a desaturated red is not affected by the Red band. If a color "
        "does not respond, it is probably too desaturated to be in the band.\n"
        "- **Use small values** → A hue shift of 10–20 degrees is usually enough. Large shifts "
        "make colors look unnatural. Always check the Vectorscope to see the effect.\n"
        "**Technical**\n"
        "Band selection: raised-cosine window of ±60° around each band center "
        "(0°, 60°, 120°, 180°, 240°, 300°); neighbouring windows sum to 1. "
        "# grade.py:_selective_color\n"
        "Hue: `h' = (h + delta_h) mod 360` in HSV. # grade.py:_selective_color\n"
        "Saturation: luminance-preserving, `out = Y + (1 + delta_s) * (x - Y)`. "
        "# grade.py:_selective_color\n"
        "Luminance: `V' = V + delta_L` in HSV. # grade.py:_selective_color\n"
        "Grey guard: `smoothstep(0.02, 0.15, S)`; full effect from S ~ 0.15, zero at S = 0. "
        "# grade.py:_selective_color\n"
        "Processing chain position: after saturation/vibrance, before split toning. "
        "Color space: HSV for hue/luminance, luma-preserving for saturation. Code: "
        "`core/refgrade/grade.py:_selective_color`.\n"
    ),
    "curves": (
        "**In short**\n"
        "- Tone curves (RGB, R, G, B) and hue curves (HvH, HvS, HvL, LvS, SvS)\n"
        "- RGB curves remap channel values; hue curves adjust per hue/luma/sat\n"
        "- All curves use PCHIP interpolation (monotone cubic)\n"
        "- S-curve: (0,0), (0.25,0.2), (0.5,0.5), (0.75,0.8), (1,1) for contrast\n"
        "- Use the clipping warning to avoid pushing channels to 0 or 1\n"
        "**Background**\n"
        "Curves are the most powerful tool in color grading. A tone curve is a graph where the "
        "horizontal axis is the input brightness (0 = black, 1 = white) and the vertical axis "
        "is the output brightness. By bending the curve, you can make the image brighter, "
        "darker, more contrasty, or add a creative tint.\n"
        "The RGB curve affects all three channels equally — it is a global tone adjustment. "
        "The R, G, and B curves affect individual channels, so you can add color: lifting the "
        "lower end of the R curve adds a warm cast to the shadows; lifting the upper end of "
        "the B curve adds a cool cast to the highlights. This is the classic \"teal and orange\" "
        "look.\n"
        "Hue curves go further: they let you adjust hue, saturation, or luminance as a function "
        "of the pixel's hue, luminance, or saturation. For example, HvS (Hue vs Saturation) "
        "lets you reduce the saturation of only the blue hues (muted sky) while leaving the "
        "red hues vivid. HvL (Hue vs Luminance) lets you brighten only the yellow hues "
        "(lighter skin tones).\n"
        "To use it: (1) Start with the RGB curve for overall tone. (2) Add an S-curve for "
        "contrast: drag the lower-left point down and the upper-right point up. (3) Use the "
        "channel curves for creative tint. (4) Use hue curves for fine per-color control. "
        "(5) Watch the clipping warning — if a channel is pushed to 0 or 1, detail is lost.\n"
        "Common mistakes: too aggressive an S-curve (losing shadow/highlight detail); not "
        "checking the clipping warning; using hue curves without understanding the axes "
        "(the Vectorscope helps: the horizontal axis is hue, the distance from center is "
        "saturation).\n"
        "In the scopes: the HIST shows the result of the RGB curve — the neon lines should "
        "span the full range without hitting the edges. The PARADE shows each channel "
        "separately — if one channel is significantly higher than the others, there is a "
        "color cast. The Vectorscope shows the effect of hue curves on the color distribution.\n"
        "Interplay: Curves is the fifth chapter in the chain (after RGB, before Wheels), so it "
        "acts on the result of the earlier chapters (all chapters are still baked into the 33³ "
        "LUT). Changing the RGB curve shifts the starting point for Wheels and HSL. Use it for "
        "the tone adjustment before the Wheels and HSL fine-tuning.\n"
        "**Practical — real scenarios**\n"
        "- **Image looks flat, needs contrast** → Problem: the photo lacks depth and punch. "
        "Solution: build an S-curve on the RGB channel — drag the lower-left point down (darker "
        "shadows) and the upper-right point up (brighter highlights). Start gentle: (0.25, 0.2) "
        "and (0.75, 0.8). Watch the clipping warning.\n"
        "- **Teal & orange look** → Problem: you want the classic cinematic grade. Solution: "
        "lift the upper end of the Red curve (warmer highlights) and the lower end of the Blue "
        "curve (cooler shadows). Keep the Green curve flat. This is the same as Shadows cyan + "
        "Highlights orange in the Wheels, but with more precision.\n"
        "- **One channel is clipping** → Problem: the clipping warning shows a channel pushed "
        "to 0 or 1, and detail is lost. Solution: pull that channel's curve back — if Red hits "
        "the top, lower the upper-right point. Check the PARADE scope to see which channel is "
        "clipping.\n"
        "- **Muted sky, vivid subject** → Problem: you want to desaturate only the blue sky "
        "while keeping the rest vivid. Solution: use the HvS (Hue vs Saturation) curve — drag "
        "the blue-hue point down to reduce its saturation, leaving the red and yellow hues "
        "alone.\n"
        "- **Lighter skin tones** → Problem: the face is a bit dark and you want to brighten "
        "only the warm hues. Solution: use the HvL (Hue vs Luminance) curve — drag the "
        "yellow/orange-hue point up to brighten the skin, leaving the blue hues (sky) alone.\n"
        "- **Use the clipping warning** → If a channel is pushed to 0 or 1, detail is lost and "
        "cannot be recovered. Keep the curve inside the safe range, especially at the edges.\n"
        "**Technical**\n"
        "RGB curves: PCHIP (Fritsch-Carlson) through user points, x strictly increasing in "
        "[0,1], y in [0,1]; constant outside the first/last point. # curves.py:apply\n"
        "Hue curves (HvH, HvS, HvL): cyclic PCHIP on the hue axis (0..1 wraps), in LCh(ab) "
        "space; chroma-weighted so grey pixels are unaffected. # huecurves.py:apply\n"
        "Hue curves (LvS, SvS): non-cyclic PCHIP, constant outside the first/last point. "
        "# huecurves.py:apply\n"
        "Identity: `RGB = [[0,0],[1,1]]`; `Hue = all y = 0.5`.\n"
        "Processing chain: curves are part of the LUT (baked with all sections). "
        "Color space: sRGB for RGB curves, LCh(ab) for hue curves. Code: "
        "`core/refgrade/curves.py`, `core/refgrade/huecurves.py`.\n"
    ),
    "wheels": (
        "**In short**\n"
        "- Highlights (Gain), Mids (Gamma), Shadows (Lift): point tints its zone\n"
        "- Direction = hue, distance from centre = strength\n"
        "- Slider under each wheel makes the zone brighter or darker\n"
        "- Blending controls zone overlap; Balance shifts the boundary\n"
        "- Teal & Orange: Shadows cyan, Highlights orange, Blending 0.2\n"
        "**Background**\n"
        "The Wheels are the most creative tool in RefGrade. Each wheel controls one tonal "
        "zone: the top wheel (Highlights) tints the bright areas, the middle wheel (Mids) "
        "tints the midtones, and the bottom wheel (Shadows) tints the dark areas. The point "
        "you place on the wheel determines the color: the direction from the center is the "
        "hue (angle), and the distance from the center is the strength. The slider under each "
        "wheel makes that zone brighter or darker, independent of the color.\n"
        "Think of it like three colored gels, one for each brightness range. Put the Shadows "
        "wheel toward blue and the Highlights wheel toward orange, and you get the classic "
        "\"teal and orange\" cinematic look. Put all three wheels slightly toward yellow and "
        "you get a warm, nostalgic film look. Keep them at the center and the image is "
        "untinted.\n"
        "Blending controls how much the zones overlap. At 0, the zones are completely separate "
        "(shadows only, mids only, highlights only). At 1, all three wheels act on all tones "
        "equally (like a global tint). The default 0.5 gives a smooth transition. Balance "
        "moves the boundary between shadows and highlights: positive means the Highlights "
        "wheel reaches further down into the midtones.\n"
        "To use it: (1) Keep all wheels at center to start. (2) Choose a creative direction: "
        "teal & orange, warm film, cool noir. (3) Drag the wheel points — small movements "
        "first. (4) Adjust Blending for the zone separation you want. (5) Use the sliders "
        "under the wheels for brightness. (6) Check the Vectorscope — the cloud should shift "
        "in the direction you expect.\n"
        "Common mistakes: too much strength (the image looks like a colored filter); "
        "opposite colors in Shadows and Highlights with high Blending (they cancel in the "
        "midtones); forgetting that the wheel color is multiplicative (it tints, it does not "
        "replace).\n"
        "In the scopes: the Vectorscope is the best tool for Wheels — it shows the color "
        "distribution as a cloud. When you tint the Shadows blue, the lower part of the cloud "
        "(dark, desaturated colors) shifts toward blue. The HIST shows the brightness effect "
        "of the wheel sliders.\n"
        "Interplay: Wheels is the sixth chapter in the chain (after Curves, before HSL), so it "
        "tints the brightness that the earlier chapters produced. It interacts with HSL (both "
        "can tint specific colors) and with the RGB section (both do per-channel adjustments). "
        "Use Wheels for creative tint, HSL for precise color correction, RGB for video-style "
        "primary correction.\n"
        "**Practical — real scenarios**\n"
        "- **Classic teal & orange** → Problem: you want the cinematic look with cool shadows "
        "and warm highlights. Solution: drag the Shadows wheel toward cyan (blue-green), the "
        "Highlights wheel toward orange (yellow-red), and set Blending to 0.2–0.3 for a clean "
        "split. Check the Vectorscope — the lower cloud shifts blue, the upper shifts orange.\n"
        "- **Warm, nostalgic film look** → Problem: you want a soft, warm, vintage feel. "
        "Solution: move all three wheels slightly toward yellow, keep them close to the center "
        "(low strength), and set Blending to 0.7–1 so the warm cast covers the whole image.\n"
        "- **Opposite colors cancel in the midtones** → Problem: you put orange in Highlights "
        "and blue in Shadows, but the midtones look muddy. Solution: lower Blending to 0–0.3 "
        "so the zones do not overlap, or move the wheels closer to the center to reduce the "
        "strength.\n"
        "- **Image looks like a colored filter** → Problem: the wheel strength is too high, "
        "the whole image is tinted. Solution: move the wheel points closer to the center "
        "(less strength), or lower the Intensity slider. The wheel color is multiplicative — "
        "it tints, it does not replace.\n"
        "- **Zone is too bright or too dark** → Problem: the tint is right, but the brightness "
        "of a zone is off. Solution: use the slider under each wheel to make that zone brighter "
        "or darker, independent of the color.\n"
        "- **Start at center** → Keep all wheels at the center to begin. Small movements first, "
        "then check the Vectorscope to see the effect before going further.\n"
        "**Technical**\n"
        "Zone masks: `p = 0.5 - 0.35 * Balance`; `w = 0.04 + 0.16 * clamp(Blending,0,0.5)*2`; "
        "`S = 1 - smoothstep(p-0.15-w, p-0.15+w, Y)`, `H = smoothstep(p+0.15-w, p+0.15+w, Y)`, "
        "`M = clamp(1-S-H)` — between the two edges Lift and Gain do nothing. Overlap: "
        "`o = (Blending-0.5).clamp(0,0.5)*2`; `S_hat = o + (1-o)*S`. # grade.py:_lgg\n"
        "Per channel: `ch = (1 + (gain-1)*H_hat) * (ch + lift*(1-ch)*S_hat)`, then "
        "`ch = ch^(1/(1+(gamma-1)*M_hat))`. # grade.py:_lgg\n"
        "Intensity k: `lift*k, gamma^k, gain^k` (scales color only, not the m-master). "
        "# grade.py:_lgg\n"
        "Master: `out = gain_m * (out + lift_m*(1-out))`, then `out = out^(1/gamma_m)`, "
        "global (no zone mask). # grade.py:_lgg\n"
        "At the wheel rim (k=1): `lift ±0.15, gamma x0.71...1.41, gain x0.76...1.32`.\n"
        "Processing chain position: after temp/tint, before contrast. Color space: sRGB with "
        "Rec.709 luma for zone masks. Code: `core/refgrade/grade.py:_lgg`.\n"
    ),
    "lut": (
        "**In short**\n"
        "- Loads a .cube 3D LUT from the folder (subfolders = categories)\n"
        "- Hover the list for a live preview\n"
        "- SAVE stores the current grading as a .cube file\n"
        "- A LUT cannot hold spatial effects (Clarity, Texture, Dehaze)\n"
        "- Exported .cube files work in any application that supports .cube\n"
        "**Background**\n"
        "A 3D LUT (Look-Up Table) is a pre-computed color transformation. Imagine a 33×33×33 "
        "grid (33 levels per channel) where each input color (R,G,B) is mapped to an output "
        "color. When you apply a LUT, the tool looks up each pixel's color in this grid and "
        "replaces it with the stored output. This is how video cameras, film emulations, and "
        "color grading suites apply complex color transformations in real time.\n"
        "In RefGrade, you can load .cube files (the standard format for 3D LUTs) from a folder. "
        "The folder can have subfolders for categories — the dropdown shows them as a tree. "
        "Hover over a LUT in the list to see a live preview without selecting it. When you "
        "select a LUT, it is applied with trilinear interpolation (smooth blending between "
        "grid points) as the last link of the chain, on the result of all other sections.\n"
        "The SAVE button does the reverse: it takes your current grading (all sections, all "
        "sliders) and bakes it into a 33³ LUT, saved as a .cube file in the LUT folder's "
        "RefGrade subfolder. This lets you capture a look you have dialled in and reload it "
        "later, or share it with others.\n"
        "To use it: (1) Choose a LUT folder (with .cube files). (2) Browse the dropdown — "
        "hover to preview. (3) Select a LUT to apply it. (4) Adjust the Amount slider for "
        "strength. (5) Combine with other sections (Wheels, HSL) for a custom look. (6) Press "
        "SAVE to store the result as a new .cube file.\n"
        "Common mistakes: expecting a LUT to hold spatial effects (Clarity, Texture, Dehaze "
        "are not in the LUT); using a LUT designed for a different color space (most .cube "
        "files are sRGB); not checking the preview before applying.\n"
        "In the scopes: after applying a LUT, check the HIST to see how the tonal distribution "
        "changed. The Vectorscope shows the color shift. The PARADE shows per-channel changes.\n"
        "Interplay: The LUT is the last chapter in the chain (after HSL), so it works on the "
        "already graded picture — your Wheels and Curves feed into it. The LUT is part of the 33³ LUT "
        "that RefGrade builds internally — it is not applied separately.\n"
        "**Practical — real scenarios**\n"
        "- **Film emulation** → Problem: you want the look of a specific film stock (e.g., "
        "Kodak Portra, Fuji Velvia). Solution: load the corresponding .cube LUT from the folder, "
        "hover over it in the dropdown to preview, then select it. Adjust the Amount slider "
        "(0.5–0.8) to blend the LUT with your manual grade.\n"
        "- **Capture a look you dialled in** → Problem: you spent time adjusting the sliders "
        "and want to save the result. Solution: press SAVE — the current grading is baked into "
        "a 33³ .cube file in the LUT folder's RefGrade subfolder. You can reload it later or "
        "share it with others.\n"
        "- **LUT does not hold spatial effects** → Problem: you saved a LUT, but Clarity, "
        "Texture, and Dehaze are not included. Solution: this is expected — spatial effects "
        "depend on neighbouring pixels and cannot be stored in a LUT. They are applied on top "
        "of the LUT at full resolution.\n"
        "- **LUT looks wrong / color space mismatch** → Problem: the LUT was designed for a "
        "different color space (most .cube files are sRGB, but some are for linear or other "
        "spaces). Solution: check the LUT's documentation. If it is not sRGB, the colors will "
        "look off. Use a LUT that is designed for sRGB.\n"
        "- **Combine LUT with Wheels/HSL** → Problem: you want a LUT look but with your own "
        "creative tint. Solution: the LUT is the last chapter, so it works on the result of "
        "Wheels and HSL. Set up your Wheels and HSL first, then apply the LUT on top.\n"
        "- **Hover to preview** → You do not need to select a LUT to see its effect. Hover "
        "over it in the dropdown for a live preview without committing.\n"
        "**Technical**\n"
        "LUT: `33x33x33` grid (LUT_SIZE = 33); trilinear interpolation in RGB space. "
        "# lut.py:apply\n"
        "The LUT is built by sampling the full grading pipeline (all sections) on a `33^3` "
        "grid of input colors. # session.py:build_lut\n"
        "Clarity, Texture, and Dehaze are spatial effects and are NOT included in the LUT.\n"
        "Color space: sRGB (0..1). Code: `core/refgrade/lut.py`, `core/refgrade/session.py`.\n"
    ),
    "rgb": (
        "**In short**\n"
        "- Per-channel Gamma, Offset, and Gain for Master, R, G, B\n"
        "- Master acts on all channels globally; R/G/B act in selected zones\n"
        "- Classic video color-correction chain\n"
        "- Use zones to limit the effect (Shadows/Mids/Highs switches)\n"
        "- Check the Parade scope: peaks at the same height = neutral\n"
        "**Background**\n"
        "The RGB section is the classic video color-correction tool, the same one used in "
        "broadcast television and film post-production. It has three controls per channel "
        "(Master, Red, Green, Blue): Gamma (midtone brightness), Offset (brightness shift for "
        "all tones), and Gain (overall brightness). The Master controls act on all three "
        "channels equally — they are global brightness adjustments. The R, G, and B controls "
        "act on individual channels, so you can add or remove color.\n"
        "Think of it like three dimmer switches (one for each color) plus a master dimmer. "
        "The Gamma control is like adjusting the middle of the dimmer's range — it affects "
        "the midtones most. The Offset control shifts the entire range up or down — it adds "
        "a flat brightness to all tones (most visible in the shadows). The Gain control "
        "scales the entire range — it makes the channel brighter or darker overall.\n"
        "The zone switches (Shadows, Mids, Highs) let you limit where the R/G/B controls act. "
        "For example, turn on only Shadows and adjust R Offset to +0.05: this adds a warm "
        "cast to the shadows only, leaving the midtones and highlights untouched. This is "
        "how video colorists do split-toning.\n"
        "To use it: (1) Start with Master Gamma for overall midtone brightness. (2) Use "
        "Master Gain for overall brightness. (3) Use R/G/B Gamma for per-channel midtone "
        "correction. (4) Use R/G/B Offset for color casts (R Offset + = warm, B Offset + = "
        "cool). (5) Use the zone switches to limit the effect. (6) Check the Parade scope — "
        "if the R, G, B peaks are at the same height, the image is neutral.\n"
        "Common mistakes: too much Offset (the image looks washed out); not using the zone "
        "switches (the effect is too global); confusing Offset with Gain (Offset shifts all "
        "tones equally, Gain scales them).\n"
        "In the scopes: the PARADE is the key scope for RGB — it shows R, G, B side by side. "
        "If the peaks are at the same height, the image is color-neutral. If R is higher, "
        "the image is warm. If B is higher, it is cool. The HIST shows the overall brightness "
        "distribution.\n"
        "Interplay: RGB is the fourth chapter in the chain (after Tone, before Curves), so it "
        "corrects the color before the Curves, Wheels and HSL fine-tuning. It is the most "
        "precise tool for color correction (baked with all sections into the 33³ LUT). The "
        "zone switches make it as flexible as the Wheels, but with numeric precision.\n"
        "**Practical — real scenarios**\n"
        "- **Warm cast in the shadows** → Problem: the dark areas have an orange tint. "
        "Solution: turn on only the Shadows zone switch, then lower R Offset to -0.03 to -0.05 "
        "(removes red from the shadows). Check the PARADE scope — the R peak in the shadows "
        "should drop to match G and B.\n"
        "- **Cool cast in the highlights** → Problem: the bright areas have a blue tint. "
        "Solution: turn on only the Highlights zone switch, then lower B Offset to -0.03 to "
        "-0.05 (removes blue from the highlights). Check the PARADE scope.\n"
        "- **Channel is too bright overall** → Problem: one channel (e.g., Green) is too "
        "bright across the whole image. Solution: lower G Gain to 0.9–0.95 (scales the whole "
        "channel). Use G Gamma if only the midtones are too bright.\n"
        "- **Blacks are not black (washed out)** → Problem: the darkest areas are grey, not "
        "black. Solution: lower M Offset to -0.02 to -0.05 (shifts all tones down), or lower "
        "M Gain to 0.95. Check the HIST — the left edge should reach zero.\n"
        "- **Use zones to limit the effect** → Problem: a global correction affects areas you "
        "do not want to change. Solution: use the zone switches (Shadows, Mids, Highs) to "
        "limit where the R/G/B controls act. For example, turn on only Shadows to correct "
        "only the dark areas.\n"
        "- **Check the PARADE scope** → The PARADE shows R, G, B side by side. If the peaks "
        "are at the same height, the image is color-neutral. If R is higher, the image is "
        "warm. If B is higher, it is cool. Use this to guide your corrections.\n"
        "**Technical**\n"
        "Master (global): `out = (ch^(1/gamma) * gain + offset)` applied to all channels. "
        "# rgb.py:rgb_grade\n"
        "Channel (zonal): same formula per R/G/B, blended with zone masks: "
        "`out = orig * (1-total) + graded * total`, where `total = use_sh*s + use_m*m + use_h*h`. "
        "# rgb.py:_apply_zones\n"
        "Zone masks: smoothstep at luminance 0.35 (shadows/mids) and 0.65 (mids/highlights); "
        "`s + m + h = 1`. # rgb.py:_zone_masks\n"
        "Luminance: `Y = 0.2126*R + 0.7152*G + 0.0722*B` (Rec.709). # rgb.py:rgb_grade\n"
        "Processing chain: RGB is part of the LUT (baked with all sections). Color space: "
        "sRGB with Rec.709 luma. Code: `core/refgrade/rgb.py`.\n"
    ),
}

UI_HELP: dict[str, str] = {
    # Reference tiles
    "tile_0": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    "tile_1": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    "tile_2": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    "tile_3": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    "tile_4": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    "tile_5": "Reference slot. Click to select, right-click for options, or drag an image to load.",
    # Method buttons
    "method_reinhard": "Reinhard method: tone-mapping-based color transfer.",
    "method_mkl": "MKL method: transfer in a perceptually uniform color space.",
    "method_zonal": "Zonal method: zone-based selective color transfer.",
    "method_pair": "Learn the grade from a before/after pair of the same picture and apply it here. First tile = before, second = after, or one side-by-side image.",
    # FIT
    "fit_btn": "Automatically adjusts the visible sliders to match the reference.",
    # Auto ("A") per section — K30c
    "auto_tone": "Auto Tone: sets Exposure and Contrast so the histogram fills the full tonal range.",
    "auto_color": "Auto Color: neutralises the colour cast (Temperature, Tint) from the near-neutral areas.",
    "auto_wheels": "Auto Wheels: neutralises the colour cast in Shadows, Mids and Highlights.",
    "auto_rgb": "Auto RGB: sets per-channel Gamma, Gain and Offset to stretch each channel to the full range.",
    "auto_curves": "Auto Curves: builds a 3-point contrast curve from the image's percentiles.",
    # Section headers (ON/OFF, reset) per section
    "sec_reference_on": "Switch this section on or off. Only ON sections are in the mix.",
    "sec_reference_undo": "Reset this section to its default values.",
    "sec_tone_on": "Switch this section on or off. Only ON sections are in the mix.",
    "sec_tone_undo": "Reset this section to its default values.",
    "sec_color_on": "Switch this section on or off. Only ON sections are in the mix.",
    "sec_color_undo": "Reset this section to its default values.",
    "sec_wheels_on": "Switch this section on or off. Only ON sections are in the mix.",
    "sec_wheels_undo": "Reset this section to its default values.",
    # Wheels
    "wheel_gain": "Highlights (Gain): tint and brightness of the bright tones.",
    "wheel_gamma": "Mids (Gamma): tint and brightness of the midtones.",
    "wheel_lift": "Shadows (Lift): tint and brightness of the dark tones.",
    # LUT section
    "sec_lut_on": "Switch this section on or off. Only ON sections are in the mix.",
    "sec_lut_undo": "Reset the LUT selection to — none — and switch the section off.",
    "lut_folder": "Choose the folder containing .cube LUT files.",
    "lut_combo": "Select a LUT to apply. Scroll to browse. — none — switches the section off.",
    "lut_save": "Save the current grading as a .cube LUT in the LUT folder's RefGrade subfolder.",
    # Amount sliders (one per section)
    "amount_reference": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_tone": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_color": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_wheels": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_hsl": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_curves": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    "amount_lut": "How strongly this section acts. 0 % = off, 100 % = full effect.",
    # MASTER
    "master": "Master strength: scales the entire grading effect.",
    # Action buttons
    "btn_reset": "Reset all sections, sliders, and wheels to their defaults.",
    "btn_ok": "Apply the grading as a new layer and close.",
    "btn_cancel": "Close without applying.",
    # Image view and histogram
    "image_view": "Image view: scroll to zoom, middle-drag to pan, F to fit, 1 for 100 %.",
    "histogram": "Luminance histogram: grey lines = source and reference, red area = result.",
    "hsl_mode_hue": "Hue: shift each colour band towards its neighbours.",
    "hsl_mode_sat": "Saturation: more or less colour per band.",
    "hsl_mode_light": "Luminance: lighter or darker per colour band.",
    "wb_auto": "Auto white balance: neutralises the cast of the near-neutral areas (plaster, concrete, sky).",
    "wb_pick": "Eyedropper: click a spot that should be neutral grey/white; Temperature and Tint follow.",
    # POP
    "tone_pop": "One-click 'make it pop': contrast, highlights/shadows, whites/blacks, vibrance and clarity. Everything except clarity goes into SAVE LUT.",
    "not_in_lut": "Clarity is local contrast: it depends on neighbouring pixels, so no LUT can hold it. It goes onto the layer as a Krita filter mask (editable later) and is left out of SAVE LUT.",
}

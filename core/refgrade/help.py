"""Help texts for the sliders (single source for tooltips and UI help).

Each entry: ``{"short": "...", "en": "...", "tip": "..."}``.
- short: ≤ 8 words, beginner language, shown bold in the tooltip.
- en: formula / technical description.
- tip: practical advice.
Range and default are appended at display time from SLIDERS.
"""

from __future__ import annotations

HELP: dict[str, dict[str, str]] = {
    # ── SLIDERS: match ─────────────────────────────────────────────────────
    "light_strength": {
        "short": "How much light mood transfers",
        "en": "Scales the Lab L* transfer: out = x + s * (matched_L - x), where s is this value.",
        "tip": "0 = no light transfer, 1 = full. Try 0.5–0.7 for very different exposures.",
    },
    "color_strength": {
        "short": "How much color mood transfers",
        "en": "Scales the Lab a*/b* transfer: out = x + s * (matched_ab - x_ab), independent of light.",
        "tip": "0 = no color transfer, 1 = full. Adjustable independently of light strength.",
    },
    "protect_shadows": {
        "short": "Keep dark areas close to original",
        "en": "Blends matched back toward original in dark zones: out = orig + (matched-orig)*(1-ps*w_sh).",
        "tip": "0.3–0.5 if shadows become too bright or too colorful.",
    },
    "protect_midtones": {
        "short": "Keep midtones close to original",
        "en": "Blends matched back toward original in mid-tone zones: out = orig + (matched-orig)*(1-pm*w_mid).",
        "tip": "1.0 = midtones stay as in the original; 0.3–0.5 only dampens.",
    },
    "protect_highlights": {
        "short": "Keep bright areas close to original",
        "en": "Blends matched back toward original in bright zones: out = orig + (matched-orig)*(1-ph*w_hi).",
        "tip": "0.3–0.5 if highlights look overexposed or unnatural.",
    },
    # ── SLIDERS: exposure ──────────────────────────────────────────────────
    "exposure": {
        "short": "Overall brightness, like camera exposure",
        "en": "Multiplies linear RGB by 2^value: x_lin = x_lin * 2^exposure, then re-encodes to sRGB.",
        "tip": "Positive = brighter, negative = darker. 1 = one stop. Typical ±0.3–0.7.",
    },
    "temperature": {
        "short": "Warm-cool shift of the whole image",
        "en": "White balance in linear light: red ×2^(0.8·t), blue ×2^(−0.8·t), grey keeps its brightness (positive = warmer/yellow, negative = cooler/blue).",
        "tip": "Small values (±0.1–0.3) are usually enough; ±0.5–1.0 for a strong creative look.",
    },
    "tint": {
        "short": "Green-magenta shift of the whole image",
        "en": "White balance in linear light: green ×2^(−0.5·tint), grey keeps its brightness (positive = magenta, negative = green).",
        "tip": "±0.1–0.3 for subtle skin toning; use AUTO WB first to kill the cast.",
    },
    # ── SLIDERS: lgg ───────────────────────────────────────────────────────
    "lift_m": {
        "short": "Master shadow brightness (all channels)",
        "en": "Master shadow lift (additive, applied before gain): out = gain_m*(out + lift_m*(1-out)); the term lift_m*(1-out) brightens shadows. Global (no zone mask).",
        "tip": "Negative for deeper blacks, positive for soft milky shadows. Range -0.5 to 0.5.",
    },
    "lift_r": {
        "short": "Red tint in shadows",
        "en": "Red channel shadow lift: ch = ch + lift_r*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = warmer shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
    },
    "lift_g": {
        "short": "Green tint in shadows",
        "en": "Green channel shadow lift: ch = ch + lift_g*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = green shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
    },
    "lift_b": {
        "short": "Blue tint in shadows",
        "en": "Blue channel shadow lift: ch = ch + lift_b*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = cooler shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
    },
    "gamma_m": {
        "short": "Master midtone brightness (all channels)",
        "en": "Master midtone gamma: out = out^(1/gamma_m), applied globally (no zone mask).",
        "tip": "1.0 = neutral. Higher = brighter, lower = darker. Range 0.2–5.0; typical 0.8–1.2.",
    },
    "gamma_r": {
        "short": "Red tint in midtones",
        "en": "Red channel midtone gamma: ch = ch^(1/(1+(gamma_r-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 brightens red in the midtones (warmer), below 1 cooler. Range 0.2–5.0; typical 0.8–1.2.",
    },
    "gamma_g": {
        "short": "Green tint in midtones",
        "en": "Green channel midtone gamma: ch = ch^(1/(1+(gamma_g-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 adds green to the midtones, below 1 pushes toward magenta. Range 0.2–5.0; typical 0.8–1.2.",
    },
    "gamma_b": {
        "short": "Blue tint in midtones",
        "en": "Blue channel midtone gamma: ch = ch^(1/(1+(gamma_b-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 adds blue to the midtones (cooler), below 1 warmer. Range 0.2–5.0; typical 0.8–1.2.",
    },
    "gain_m": {
        "short": "Master highlight brightness (all channels)",
        "en": "Master highlight gain (multiplicative): out = gain_m*(out + lift_m*(1-out)); gain_m scales the whole lifted result. Global (no zone mask).",
        "tip": "1.0 = neutral. Higher = brighter highlights. Range 0–4.0; typical 0.8–1.3.",
    },
    "gain_r": {
        "short": "Red tint in highlights",
        "en": "Red channel highlight gain: ch = (1+(gain_r-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 brightens red in the highlights (warmer), below 1 cooler. Range 0–4.0; typical 0.8–1.3.",
    },
    "gain_g": {
        "short": "Green tint in highlights",
        "en": "Green channel highlight gain: ch = (1+(gain_g-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 adds green to the highlights, below 1 pushes toward magenta. Range 0–4.0; typical 0.8–1.3.",
    },
    "gain_b": {
        "short": "Blue tint in highlights",
        "en": "Blue channel highlight gain: ch = (1+(gain_b-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 adds blue to the highlights (cooler), below 1 warmer. Range 0–4.0; typical 0.8–1.3.",
    },
    "wheel_blending": {
        "short": "How much the three wheels overlap",
        "en": "How much the Shadows, Mids and Highlights wheels overlap (like Lightroom's Blending). "
              "The split sits at luminance p = 0.5 − 0.35·Balance; a smoothstep of half-width w "
              "fades Shadows out and Highlights in around it. 0…0.5 widens w from 0.10 (pure zones) "
              "to 0.40; 0.5…1 adds overlap until every wheel acts on every tone (1 = classic "
              "lift/gamma/gain).",
        "tip": "Default 0.5. Opposite colours (orange highlights, blue shadows) cancel in the "
               "midtones when the zones overlap: go to 0–0.3 for a clean split-tone, 0.7–1 for a "
               "soft overall cast.",
    },
    "wheel_balance": {
        "short": "Moves the shadow/highlight boundary",
        "en": "Moves the split between shadows and highlights: p = 0.5 − 0.35·Balance on the "
              "luminance scale 0…1.",
        "tip": "Positive = the highlight wheel reaches further down into the midtones, negative = "
               "the shadow wheel reaches further up.",
    },
    "wheel_intensity": {
        "short": "Strength of all three colour wheels",
        "en": "Reach of all three colour wheels. Scales only their colour, not their brightness "
              "(H/M/L master): lift·k, gamma^k, gain^k. At the wheel's rim and k = 1 that is lift "
              "±0.15, gamma ×0.71…1.41, gain ×0.76…1.32; k = 2 doubles the lift and squares the "
              "factors.",
        "tip": "1 = normal. Raise it when a wheel at the rim is still too weak; lower it for fine "
               "work near the centre. Beyond ~1.5 bright tones clip quickly.",
    },
    # ── SLIDERS: contrast ──────────────────────────────────────────────────
    "contrast": {
        "short": "Difference between light and dark",
        "en": "Scales around pivot: y = (x - pivot) * contrast + pivot, with smooth rolloff above luma 0.7.",
        "tip": "1.0 = neutral. 1.1–1.3 for more punch, 0.8–0.9 for softer.",
    },
    "pivot": {
        "short": "Where contrast pivots (neutral point)",
        "en": "Neutral point of the contrast formula y = (x - pivot) * contrast + pivot (0 = black, 1 = white).",
        "tip": "0.5 = center. Move up to protect shadows, down to protect highlights.",
    },
    "highlight_rolloff": {
        "short": "Softens the top to prevent clipping",
        "en": "Soft rolloff above luma 0.7: y *= 1 - rolloff * 0.5 * smoothstep(0.7, 1.0, Y) prevents hard clipping.",
        "tip": "0.2–0.4 for natural highlights without a hard edge. 0 = no rolloff.",
    },
    "highlights": {
        "short": "Recovers or pushes the brightest areas",
        "en": "Adds a Gaussian bump at L* 0.75: L' = L + 0.25 * hi * bump(0.75), monotone-enforced.",
        "tip": "Negative (-0.3 to -0.5) recovers blown highlights; positive brightens the top end.",
    },
    "shadows": {
        "short": "Lifts or deepens the darkest areas",
        "en": "Adds a Gaussian bump at L* 0.25: L' = L + 0.25 * sh * bump(0.25), monotone-enforced.",
        "tip": "Positive (+0.3) lifts dark areas; negative deepens them.",
    },
    "whites": {
        "short": "Shifts the white point up or down",
        "en": "Shifts the white point: L' = L + 0.25 * wh * L^3, pushing the brightest pixels toward white.",
        "tip": "Positive pushes highlights to white, negative pulls them back. Typical ±0.1–0.3.",
    },
    "blacks": {
        "short": "Shifts the black point up or down",
        "en": "Shifts the black point: L' = L + 0.25 * bl * (1-L)^3, pushing the darkest pixels toward black.",
        "tip": "Negative deepens blacks, positive lifts them. Typical ±0.1–0.3.",
    },
    "clarity": {
        "short": "Micro-contrast in midtones (spatial)",
        "en": "Gaussian unsharp on L* only, masked to midtones: amount = 0.6 * clarity. Not in LUT.",
        "tip": "+0.2–0.4 adds micro-contrast; negative softens skin. Spatial effect, not in LUT.",
    },
    "texture": {
        "short": "Fine detail sharpness (spatial)",
        "en": "High-pass (Gaussian, sigma ~ 2 px at 1024) on L*: amount = 1.5 * texture. Not in LUT.",
        "tip": "+0.2–0.4 sharpens fine details; negative smooths. Spatial effect, not in LUT.",
    },
    "dehaze": {
        "short": "Removes or adds atmospheric haze (spatial)",
        "en": "Dark Channel Prior (He et al. 2009): estimates and removes atmospheric light. Not in LUT.",
        "tip": "+0.2–0.4 removes haze; negative adds haze. Spatial effect, not in LUT.",
    },
    # ── SLIDERS: color ─────────────────────────────────────────────────────
    "saturation": {
        "short": "Overall color intensity",
        "en": "Scales color around luma: out = Y + s * (x - Y), where Y = 0.2126R + 0.7152G + 0.0722B.",
        "tip": "1.0 = neutral. 0.8–0.9 for subtler, 1.1–1.3 for more vivid.",
    },
    "vibrance": {
        "short": "Boosts only pale colors, spares vivid ones",
        "en": "Saturation weighted by 1-sat: factor = 1 + v*(1-sat), out = Y + factor*(x-Y); spares saturated colors.",
        "tip": "+0.2–0.4 adds color in midtones without oversaturating; negative desaturates.",
    },
    # ── SLIDERS: sel (selective color) ─────────────────────────────────────
    "sel_red_hue": {
        "short": "Shifts red hue toward orange or magenta",
        "en": "Shifts hue of red band (0°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Hue grows from red (0°) toward yellow (60°): -10 to -20 moves skin from orange toward true red. Range -30 to +30 degrees.",
    },
    "sel_red_sat": {
        "short": "More or less red intensity",
        "en": "Luminance-preserving saturation for red band: out = Y + (1+delta_s)*(x-Y), cosine window.",
        "tip": "Negative for matte, positive for vivid reds. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_red_light": {
        "short": "Lighter or darker red areas",
        "en": "Luminance shift for red band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1–0.2 lightens skin tones. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_yellow_hue": {
        "short": "Shifts yellow hue toward green or orange",
        "en": "Shifts hue of yellow band (60°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Yellow sits at 60°: +10 shifts toward green, -10 toward orange. Range -30 to +30 degrees.",
    },
    "sel_yellow_sat": {
        "short": "More or less yellow intensity",
        "en": "Luminance-preserving saturation for yellow band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vibrant yellows. Range -1 to +1.",
    },
    "sel_yellow_light": {
        "short": "Lighter or darker yellow areas",
        "en": "Luminance shift for yellow band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1–0.2 lightens skin tones in yellow areas. Range -1 to +1.",
    },
    "sel_green_hue": {
        "short": "Shifts green hue toward cyan or yellow",
        "en": "Shifts hue of green band (120°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "+10 to +20 shifts toward cyan for more natural vegetation. Range -30 to +30.",
    },
    "sel_green_sat": {
        "short": "More or less green intensity",
        "en": "Luminance-preserving saturation for green band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid greens. Range -1 to +1.",
    },
    "sel_green_light": {
        "short": "Lighter or darker green areas",
        "en": "Luminance shift for green band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1 lightens vegetation. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_cyan_hue": {
        "short": "Shifts cyan hue toward blue or green",
        "en": "Shifts hue of cyan band (180°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward blue (+10) or green (-10). Range -30 to +30 degrees.",
    },
    "sel_cyan_sat": {
        "short": "More or less cyan intensity",
        "en": "Luminance-preserving saturation for cyan band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid cyans. Range -1 to +1.",
    },
    "sel_cyan_light": {
        "short": "Lighter or darker cyan areas",
        "en": "Luminance shift for cyan band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust sky brightness in cyan areas. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_blue_hue": {
        "short": "Shifts blue hue toward cyan or violet",
        "en": "Shifts hue of blue band (240°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward cyan (-10) or violet (+10). Range -30 to +30 degrees.",
    },
    "sel_blue_sat": {
        "short": "More or less blue intensity",
        "en": "Luminance-preserving saturation for blue band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "-0.2 mutes sky for a filmic look. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_blue_light": {
        "short": "Lighter or darker blue areas",
        "en": "Luminance shift for blue band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust sky brightness in blue areas. Range -1 to +1; typical ±0.1–0.3.",
    },
    "sel_magenta_hue": {
        "short": "Shifts magenta hue toward red or blue",
        "en": "Shifts hue of magenta band (300°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward red (+10) to protect lips. Range -30 to +30 degrees.",
    },
    "sel_magenta_sat": {
        "short": "More or less magenta intensity",
        "en": "Luminance-preserving saturation for magenta band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid magentas. Range -1 to +1.",
    },
    "sel_magenta_light": {
        "short": "Lighter or darker magenta areas",
        "en": "Luminance shift for magenta band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust skin tones in magenta areas. Range -1 to +1; typical ±0.1–0.3.",
    },
    # ── SLIDERS: split ─────────────────────────────────────────────────────
    "shadow_hue": {
        "short": "Color direction of the shadow tint",
        "en": "Hue angle for shadow tint in Lab: ab += amount*30*(cos(hue), sin(hue)), weighted by shadow zone.",
        "tip": "30 = warm/orange, 220 = cool/blue. Range 0–360; affects only the shadow zone.",
    },
    "shadow_amount": {
        "short": "Strength of the shadow tint",
        "en": "Strength of the Lab a*/b* offset in shadows: ab_offset = amount * 30 * (cos h, sin h).",
        "tip": "0 = no mix, 1 = full. 0.1–0.3 for a subtle effect.",
    },
    "highlight_hue": {
        "short": "Color direction of the highlight tint",
        "en": "Hue angle for highlight tint in Lab: ab += amount*30*(cos(hue), sin(hue)), weighted by highlight zone.",
        "tip": "40 = warm/golden, 200 = cool/blue. Range 0–360; affects only the highlight zone.",
    },
    "highlight_amount": {
        "short": "Strength of the highlight tint",
        "en": "Strength of the Lab a*/b* offset in highlights: ab_offset = amount * 30 * (cos h, sin h).",
        "tip": "0 = no mix, 1 = full. 0.1–0.3 for a subtle effect.",
    },
    # ── SLIDERS: rgb ───────────────────────────────────────────────────────
    "rgb_m_gamma": {
        "short": "Master midtone brightness (all channels)",
        "en": "Master gamma (global): out = ch^(1/gamma), applied to all channels before per-channel controls.",
        "tip": "1.0 = no change. 0.9 darkens, 1.1 brightens midtones. Range 0.2–5.0.",
    },
    "rgb_m_pedestal": {
        "short": "Master brightness offset (all channels)",
        "en": "Master offset (global): out = out + offset, added to all channels after gamma and gain.",
        "tip": "Positive lifts blacks, negative darkens. Range -0.5 to 0.5; typical ±0.02–0.1.",
    },
    "rgb_m_gain": {
        "short": "Master overall brightness (all channels)",
        "en": "Master gain (global): out = out * gain, multiplied on all channels after gamma.",
        "tip": "1.0 = no change. >1 brightens, <1 darkens. Range 0–4.0; typical 0.8–1.3.",
    },
    "rgb_r_gamma": {
        "short": "Red midtone brightness (zonal)",
        "en": "Red channel gamma (zonal): out = ch^(1/gamma), blended with zone masks (shadows/mids/highlights).",
        "tip": "1.0 = no change. <1 darkens reds, >1 brightens. Range 0.2–5.0.",
    },
    "rgb_r_pedestal": {
        "short": "Red brightness offset (zonal)",
        "en": "Red channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds red (warm), negative removes it. Range -0.5 to 0.5.",
    },
    "rgb_r_gain": {
        "short": "Red overall brightness (zonal)",
        "en": "Red channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts reds, <1 reduces. Range 0–4.0.",
    },
    "rgb_g_gamma": {
        "short": "Green midtone brightness (zonal)",
        "en": "Green channel gamma (zonal): out = ch^(1/gamma), blended with zone masks.",
        "tip": "1.0 = no change. <1 darkens greens, >1 brightens. Range 0.2–5.0.",
    },
    "rgb_g_pedestal": {
        "short": "Green brightness offset (zonal)",
        "en": "Green channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds green, negative removes it. Range -0.5 to 0.5.",
    },
    "rgb_g_gain": {
        "short": "Green overall brightness (zonal)",
        "en": "Green channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts greens, <1 reduces. Range 0–4.0.",
    },
    "rgb_b_gamma": {
        "short": "Blue midtone brightness (zonal)",
        "en": "Blue channel gamma (zonal): out = ch^(1/gamma), blended with zone masks.",
        "tip": "1.0 = no change. <1 darkens blues, >1 brightens. Range 0.2–5.0.",
    },
    "rgb_b_pedestal": {
        "short": "Blue brightness offset (zonal)",
        "en": "Blue channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds blue (cool), negative removes it. Range -0.5 to 0.5.",
    },
    "rgb_b_gain": {
        "short": "Blue overall brightness (zonal)",
        "en": "Blue channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts blues, <1 reduces. Range 0–4.0.",
    },
    "rgb_use_shadows": {
        "short": "Enable R/G/B effect in shadows",
        "en": "Scales the shadow zone mask (smoothstep 0–0.35 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in shadows, 0 = off. Master always acts everywhere.",
    },
    "rgb_use_midtones": {
        "short": "Enable R/G/B effect in midtones",
        "en": "Scales the midtone zone mask (smoothstep 0.35–0.65 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in midtones, 0 = off. Master always acts everywhere.",
    },
    "rgb_use_highlights": {
        "short": "Enable R/G/B effect in highlights",
        "en": "Scales the highlight zone mask (smoothstep 0.65–1.0 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in highlights, 0 = off. Master always acts everywhere.",
    },
}

# One short "Auto" paragraph per AUTO section (the "A" button in the header) — AUTO redesign
# 2026-10-13, core/refgrade/auto.py; measured in reports/auto2_fivek.md. WHEELS/RGB have no AUTO.
AUTO_HELP: dict[str, str] = {
    "tone": (
        "**Auto**\n"
        "Sets Exposure, Contrast, Highlights, Shadows, Whites and Blacks so the brightness of the "
        "picture lands where photo editors put it: seven brightness percentiles inside the range "
        "measured on expert retouches (MIT-Adobe FiveK). A picture already inside that range is left "
        "alone. It works on the picture as it arrives at TONE in the chain. Limit: a room full of dark "
        "material can be lifted too much — undo with ↺. # auto.py:_auto_tone, auto.py:BAND_LO/BAND_HI \n"
    ),
    "color": (
        "**Auto**\n"
        "Removes a colour cast: Temperature and Tint are set until the near-neutral pixels "
        "(plaster, concrete, paper) are neutral — down to a slight warmth of 2.5 (Lab a*/b*), which "
        "experts usually keep. A neutral picture is left alone. # auto.py:_auto_color, auto.py:CAST_OK \n"
    ),
    "curves": (
        "**Auto**\n"
        "A master curve that moves each brightness percentile to the nearest point of the expert range "
        "(same range as TONE). If TONE AUTO ran first, CURVES has almost nothing left to do — AUTO never "
        "corrects twice. # auto.py:_auto_curves \n"
    ),
}

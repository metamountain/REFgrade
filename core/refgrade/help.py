"""Help texts for the sliders (single source for tooltips and UI help).

Each entry: ``{"short": "...", "en": "...", "tip": "...", "what": "..."}``.
- short: ≤ 8 words, beginner language, shown bold in the tooltip.
- en: formula / technical description.
- tip: practical advice.
- what: 1–3 plain-English sentences, no formulas, what the slider does to the image
  when you drag it left/right (K40a, 2026-10-16). Shown in the per-value "?" popup.
Range and default are appended at display time from SLIDERS.
"""

from __future__ import annotations

HELP: dict[str, dict[str, str]] = {
    # ── SLIDERS: match ─────────────────────────────────────────────────────
    "light_strength": {
        "short": "How much light mood transfers",
        "en": "Scales the Lab L* transfer: out = x + s * (matched_L - x), where s is this value.",
        "tip": "0 = no light transfer, 1 = full. Try 0.5–0.7 for very different exposures.",
        "what": "Controls how strongly the brightness mood of your reference photo is copied onto the image. Right transfers more of the reference light, left keeps the original brightness. Use 0.5 to 0.7 when the two photos were taken at very different exposures.",
    },
    "color_strength": {
        "short": "How much color mood transfers",
        "en": "Scales the Lab a*/b* transfer: out = x + s * (matched_ab - x_ab), independent of light.",
        "tip": "0 = no color transfer, 1 = full. Adjustable independently of light strength.",
        "what": "Controls how strongly the color mood of your reference photo is copied onto the image. Right copies more of the reference colors, left keeps the original colors. It works independently of the light transfer.",
    },
    "protect_shadows": {
        "short": "Keep dark areas close to original",
        "en": "Blends matched back toward original in dark zones: out = orig + (matched-orig)*(1-ps*w_sh).",
        "tip": "0.3–0.5 if shadows become too bright or too colorful.",
        "what": "Keeps the dark areas of your image close to the original while the reference mood is applied. Right protects the shadows more, so they do not become too bright or too colorful. Use 0.3 to 0.5 when the shadows look off.",
    },
    "protect_midtones": {
        "short": "Keep midtones close to original",
        "en": "Blends matched back toward original in mid-tone zones: out = orig + (matched-orig)*(1-pm*w_mid).",
        "tip": "1.0 = midtones stay as in the original; 0.3–0.5 only dampens.",
        "what": "Keeps the middle brightness areas close to the original while the reference mood is applied. Right keeps the midtones as they are, left lets the reference change them more. 1.0 means the midtones never change.",
    },
    "protect_highlights": {
        "short": "Keep bright areas close to original",
        "en": "Blends matched back toward original in bright zones: out = orig + (matched-orig)*(1-ph*w_hi).",
        "tip": "0.3–0.5 if highlights look overexposed or unnatural.",
        "what": "Keeps the bright areas of your image close to the original while the reference mood is applied. Right protects the highlights more, so they do not look overexposed or unnatural. Use 0.3 to 0.5 when the bright areas look off.",
    },
    # ── SLIDERS: exposure ──────────────────────────────────────────────────
    "exposure": {
        "short": "Overall brightness, like camera exposure",
        "en": "Multiplies linear RGB by 2^value: x_lin = x_lin * 2^exposure, then re-encodes to sRGB.",
        "tip": "Positive = brighter, negative = darker. 1 = one stop. Typical ±0.3–0.7.",
        "what": "Makes the whole image brighter or darker, like opening or closing the camera shutter. Right brightens the picture, left darkens it. One step is a strong change, so 0.3 to 0.7 is usually enough.",
    },
    "temperature": {
        "short": "Warm-cool shift of the whole image",
        "en": "White balance in linear light: red ×2^(0.8·t), blue ×2^(−0.8·t), grey keeps its brightness (positive = warmer/yellow, negative = cooler/blue).",
        "tip": "Small values (±0.1–0.3) are usually enough; ±0.5–1.0 for a strong creative look.",
        "what": "Shifts the whole image toward warm yellow or cool blue, like changing the white balance. Right makes it warmer, left makes it cooler. Small values of 0.1 to 0.3 are usually enough for a natural look.",
    },
    "tint": {
        "short": "Green-magenta shift of the whole image",
        "en": "White balance in linear light: green ×2^(−0.5·tint), grey keeps its brightness (positive = magenta, negative = green).",
        "tip": "±0.1–0.3 for subtle skin toning; use AUTO WB first to kill the cast.",
        "what": "Shifts the whole image toward green or magenta, like fine-tuning the white balance. Right pushes toward magenta, left toward green. Use 0.1 to 0.3 for subtle skin toning.",
    },
    # ── SLIDERS: lgg ───────────────────────────────────────────────────────
    "lift_m": {
        "short": "Master shadow brightness (all channels)",
        "en": "Master shadow lift (additive, applied before gain): out = gain_m*(out + lift_m*(1-out)); the term lift_m*(1-out) brightens shadows. Global (no zone mask).",
        "tip": "Negative for deeper blacks, positive for soft milky shadows. Range -0.5 to 0.5.",
        "what": "Lifts or deepens the shadows of all three color channels together. Right brightens the dark areas, left makes them deeper. A small positive value gives soft, milky shadows.",
    },
    "lift_r": {
        "short": "Red tint in shadows",
        "en": "Red channel shadow lift: ch = ch + lift_r*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = warmer shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
        "what": "Adds or removes red in the shadow areas. Right warms the shadows by adding red, left cools them. A typical value is 0.05 to 0.15 for a subtle warm shadow tint.",
    },
    "lift_g": {
        "short": "Green tint in shadows",
        "en": "Green channel shadow lift: ch = ch + lift_g*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = green shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
        "what": "Adds or removes green in the shadow areas. Right pushes the shadows toward green, left pulls them away. A typical value is 0.05 to 0.15 for a subtle green cast.",
    },
    "lift_b": {
        "short": "Blue tint in shadows",
        "en": "Blue channel shadow lift: ch = ch + lift_b*(1-ch)*S_hat, where S_hat is the shadow zone mask.",
        "tip": "Positive = cooler shadows. Range -0.5 to 0.5; typical ±0.05–0.15.",
        "what": "Adds or removes blue in the shadow areas. Right cools the shadows by adding blue, left warms them. A typical value is 0.05 to 0.15 for a subtle cool shadow tint.",
    },
    "gamma_m": {
        "short": "Master midtone brightness (all channels)",
        "en": "Master midtone gamma: out = out^(1/gamma_m), applied globally (no zone mask).",
        "tip": "1.0 = neutral. Higher = brighter, lower = darker. Range 0.2–5.0; typical 0.8–1.2.",
        "what": "Brightens or darkens the middle tones of all three color channels together. Right brightens the midtones, left darkens them. 1.0 is neutral, 0.8 to 1.2 is a typical range.",
    },
    "gamma_r": {
        "short": "Red tint in midtones",
        "en": "Red channel midtone gamma: ch = ch^(1/(1+(gamma_r-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 brightens red in the midtones (warmer), below 1 cooler. Range 0.2–5.0; typical 0.8–1.2.",
        "what": "Brightens or darkens the red in the midtones. Above 1.0 the midtones get warmer with more red, below 1.0 they get cooler. 1.0 is neutral.",
    },
    "gamma_g": {
        "short": "Green tint in midtones",
        "en": "Green channel midtone gamma: ch = ch^(1/(1+(gamma_g-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 adds green to the midtones, below 1 pushes toward magenta. Range 0.2–5.0; typical 0.8–1.2.",
        "what": "Brightens or darkens the green in the midtones. Above 1.0 the midtones get more green, below 1.0 they shift toward magenta. 1.0 is neutral.",
    },
    "gamma_b": {
        "short": "Blue tint in midtones",
        "en": "Blue channel midtone gamma: ch = ch^(1/(1+(gamma_b-1)*M_hat)), where M_hat is the midtone mask.",
        "tip": "1.0 = neutral; above 1 adds blue to the midtones (cooler), below 1 warmer. Range 0.2–5.0; typical 0.8–1.2.",
        "what": "Brightens or darkens the blue in the midtones. Above 1.0 the midtones get cooler with more blue, below 1.0 they get warmer. 1.0 is neutral.",
    },
    "gain_m": {
        "short": "Master highlight brightness (all channels)",
        "en": "Master highlight gain (multiplicative): out = gain_m*(out + lift_m*(1-out)); gain_m scales the whole lifted result. Global (no zone mask).",
        "tip": "1.0 = neutral. Higher = brighter highlights. Range 0–4.0; typical 0.8–1.3.",
        "what": "Brightens or darkens the highlights of all three color channels together. Right brightens the bright areas, left darkens them. 1.0 is neutral, 0.8 to 1.3 is a typical range.",
    },
    "gain_r": {
        "short": "Red tint in highlights",
        "en": "Red channel highlight gain: ch = (1+(gain_r-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 brightens red in the highlights (warmer), below 1 cooler. Range 0–4.0; typical 0.8–1.3.",
        "what": "Boosts or reduces red in the highlight areas. Above 1.0 the highlights get warmer with more red, below 1.0 they get cooler. 1.0 is neutral.",
    },
    "gain_g": {
        "short": "Green tint in highlights",
        "en": "Green channel highlight gain: ch = (1+(gain_g-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 adds green to the highlights, below 1 pushes toward magenta. Range 0–4.0; typical 0.8–1.3.",
        "what": "Boosts or reduces green in the highlight areas. Above 1.0 the highlights get more green, below 1.0 they shift toward magenta. 1.0 is neutral.",
    },
    "gain_b": {
        "short": "Blue tint in highlights",
        "en": "Blue channel highlight gain: ch = (1+(gain_b-1)*H_hat)*ch, where H_hat is the highlight mask.",
        "tip": "1.0 = neutral; above 1 adds blue to the highlights (cooler), below 1 warmer. Range 0–4.0; typical 0.8–1.3.",
        "what": "Boosts or reduces blue in the highlight areas. Above 1.0 the highlights get cooler with more blue, below 1.0 they get warmer. 1.0 is neutral.",
    },
    "wheel_gain": {
        "short": "Size of each wheel's tone zone",
        "en": "Each wheel acts on a small white area (core) at its end of the tone scale — shadows "
              "at 0, mids at 0.5, highlights at 1. Gain sizes that core: 0.1 (default) is a small "
              "area, 0.5 grows the shadow core to the whole lower half and the highlight core to "
              "the whole upper half. At the default there are big gaps between the zones where no "
              "wheel acts; raising Gain (and/or Falloff) closes them until the zones meet.",
        "tip": "Default 0.1. Raise it when a wheel should reach further into the neighbouring "
               "tones (e.g. the shadow wheel into the mids); lower it for tight, local tints. At 0 "
               "a wheel acts on almost nothing.",
        "what": "How big the tone area each wheel covers. Right makes the shadow, midtone and highlight zones grow until they touch; left shrinks them back to small areas with big gaps between them.",
    },
    "wheel_falloff": {
        "short": "Softness of the zone edges",
        "en": "How far the white core of each zone blurs into the neighbouring tones (a glow). 0 is "
              "a hard edge, 0.5 a very wide soft gradient. Together with Gain it closes the gaps "
              "between the zones: Gain grows the cores, Falloff widens the blur around them. At "
              "the default (0.1 / 0.1) the zones are separate with big gaps.",
        "tip": "Default 0.1. Raise it for a soft, gradual transition between the wheel zones; keep "
               "it low when opposite colours (orange highlights, blue shadows) must not muddy the "
               "midtones.",
        "what": "How soft the edge of each wheel zone is. Right blurs the zone into the neighbouring tones like a glow; left keeps a hard edge.",
    },
    "wheel_intensity": {
        "short": "Strength of all three colour wheels",
        "en": "Reach of all three colour wheels. Scales only their colour, not their brightness "
              "(H/M/L master): lift·k, gamma^k, gain^k. At the wheel's rim and k = 1 that is lift "
              "±0.15, gamma ×0.71…1.41, gain ×0.76…1.32; k = 2 doubles the lift and squares the "
              "factors, which quickly pushes saturated tones into an unnatural, neon look.",
        "tip": "1 = normal. Raise it when a wheel at the rim is still too weak; lower it for fine "
               "work near the centre. Beyond ~1.5 bright tones clip quickly and the tint can turn "
               "neon/oversaturated — if it looks artificial, drop it back toward 1.",
        "what": "Scales the strength of all three color wheels together. Right makes the wheel colors stronger, left weaker, without changing the brightness wheels. 1 is normal; raise it when a wheel at the rim is still too weak. Past about 1.5 the colors can look oversaturated and neon — if that happens, lower it back toward 1.",
    },
    # ── SLIDERS: contrast ──────────────────────────────────────────────────
    "contrast": {
        "short": "Difference between light and dark",
        "en": "Scales around pivot: y = (x - pivot) * contrast + pivot, with smooth rolloff above luma 0.7.",
        "tip": "1.0 = neutral. 1.1–1.3 for more punch, 0.8–0.9 for softer.",
        "what": "Increases or decreases the difference between the light and the dark areas. Right adds punch and depth, left softens the image. 1.0 is neutral, 1.1 to 1.3 for more punch.",
    },
    "pivot": {
        "short": "Where contrast pivots (neutral point)",
        "en": "Neutral point of the contrast formula y = (x - pivot) * contrast + pivot (0 = black, 1 = white).",
        "tip": "0.5 = center. Move up to protect shadows, down to protect highlights.",
        "what": "Chooses the brightness level that stays unchanged while contrast is applied. Move it up to protect the shadows, down to protect the highlights. 0.5 is the center.",
    },
    "highlight_rolloff": {
        "short": "Brings back highlights pushed past white",
        "en": "Knee k = 1 - 0.6*r. Luma Y <= k is untouched; above k, Yn = k + (1-k)*tanh((Y-k)/(1-k)) "
              "and the pixel is scaled by Yn/Y so its hue is preserved. r = 0 leaves the image bit-identical.",
        "tip": "Softly brings back highlights that your other settings pushed past white "
               "(exposure, contrast, wheels). It cannot restore detail that was already pure white in the "
               "original image. 0 = off.",
        "what": "Softly brings back highlights that your other settings pushed past pure white. Right recovers more of the blown-out bright areas, left does less. It cannot restore detail that was already pure white in the original image.",
    },
    "highlight_reconstruct": {
        "short": "Rebuilds clipped colour channels",
        "en": "Where only one or two colour channels are clipped (e.g. a red sky turned flat), "
               "rebuilds them from the colours around. Pure white areas stay white.",
        "tip": "0 = off. Spatial effect, not in LUT.",
        "what": "Rebuilds color in areas where only one or two color channels were clipped to white, like a red sky that turned flat. Right restores more color in washed-out highlights. Pure white areas stay white.",
    },
    "highlights": {
        "short": "Recovers or pushes the brightest areas",
        "en": "Adds a Gaussian bump at L* 0.75: L' = L + 0.25 * hi * bump(0.75), monotone-enforced.",
        "tip": "Negative (-0.3 to -0.5) recovers blown highlights; positive brightens the top end.",
        "what": "Brightens or darkens only the bright parts of the image. Negative values recover detail in blown highlights, positive values brighten the top end. -0.3 to -0.5 for recovery.",
    },
    "shadows": {
        "short": "Lifts or deepens the darkest areas",
        "en": "Adds a Gaussian bump at L* 0.25: L' = L + 0.25 * sh * bump(0.25), monotone-enforced.",
        "tip": "Positive (+0.3) lifts dark areas; negative deepens them.",
        "what": "Brightens or darkens only the dark parts of the image. Right lifts shadows to show more detail in dark areas, left makes them deeper. Bright areas stay as they are.",
    },
    "whites": {
        "short": "Shifts the white point up or down",
        "en": "Shifts the white point: L' = L + 0.25 * wh * L^3, pushing the brightest pixels toward white.",
        "tip": "Positive pushes highlights to white, negative pulls them back. Typical ±0.1–0.3.",
        "what": "Pushes the brightest areas toward pure white or pulls them back. Right makes the highlights brighter and cleaner, left pulls them down. Typical values are 0.1 to 0.3.",
    },
    "blacks": {
        "short": "Shifts the black point up or down",
        "en": "Shifts the black point: L' = L + 0.25 * bl * (1-L)^3, pushing the darkest pixels toward black.",
        "tip": "Negative deepens blacks, positive lifts them. Typical ±0.1–0.3.",
        "what": "Pushes the darkest areas toward pure black or lifts them. Right lifts the blacks to a softer grey, left deepens them to true black. Typical values are 0.1 to 0.3.",
    },
    "clarity": {
        "short": "Micro-contrast in midtones (spatial)",
        "en": "Gaussian unsharp on L* only, masked to midtones: amount = 0.6 * clarity. Not in LUT.",
        "tip": "+0.2–0.4 adds micro-contrast; negative softens skin. Spatial effect, not in LUT.",
        "what": "Adds or removes micro-contrast in the midtones, so edges and structure stand out more or less. Right adds a crisp, defined look, left softens it. 0.2 to 0.4 for more definition, negative to soften skin.",
    },
    "texture": {
        "short": "Fine detail sharpness (spatial)",
        "en": "High-pass (Gaussian, sigma ~ 2 px at 1024) on L*: amount = 1.5 * texture. Not in LUT.",
        "tip": "+0.2–0.4 sharpens fine details; negative smooths. Spatial effect, not in LUT.",
        "what": "Sharpens or smooths fine details across the image. Right makes small details crisper, left smooths them. 0.2 to 0.4 for sharper detail, negative to smooth.",
    },
    "dehaze": {
        "short": "Removes or adds atmospheric haze (spatial)",
        "en": "Dark Channel Prior (He et al. 2009): estimates and removes atmospheric light. Not in LUT.",
        "tip": "+0.2–0.4 removes haze; negative adds haze. Spatial effect, not in LUT.",
        "what": "Removes or adds the milky look of atmospheric haze. Right makes distant, hazy scenes clearer and darker, left adds haze. 0.2 to 0.4 for a clear look.",
    },
    # ── SLIDERS: color ─────────────────────────────────────────────────────
    "saturation": {
        "short": "Overall color intensity",
        "en": "Scales color around luma: out = Y + s * (x - Y), where Y = 0.2126R + 0.7152G + 0.0722B.",
        "tip": "1.0 = neutral. 0.8–0.9 for subtler, 1.1–1.3 for more vivid.",
        "what": "Makes all colors more or less intense. Right makes the colors more vivid, left more muted. 1.0 is neutral, 0.8 to 0.9 for subtler, 1.1 to 1.3 for more vivid.",
    },
    "vibrance": {
        "short": "Boosts only pale colors, spares vivid ones",
        "en": "Saturation weighted by 1-sat: factor = 1 + v*(1-sat), out = Y + factor*(x-Y); spares saturated colors.",
        "tip": "+0.2–0.4 adds color in midtones without oversaturating; negative desaturates.",
        "what": "Boosts only the pale colors and leaves the vivid ones alone. Right adds color to muted areas without oversaturating the loud ones, left desaturates. 0.2 to 0.4 is a typical boost.",
    },
    # ── SLIDERS: sel (selective color) ─────────────────────────────────────
    "sel_red_hue": {
        "short": "Shifts red hue toward orange or magenta",
        "en": "Shifts hue of red band (0°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Hue grows from red (0°) toward yellow (60°): -10 to -20 moves skin from orange toward true red. Range -30 to +30 degrees.",
        "what": "Shifts the hue of red areas toward orange or toward magenta. Right moves red toward magenta, left toward orange. -10 to -20 moves skin from orange toward true red.",
    },
    "sel_red_sat": {
        "short": "More or less red intensity",
        "en": "Luminance-preserving saturation for red band: out = Y + (1+delta_s)*(x-Y), cosine window.",
        "tip": "Negative for matte, positive for vivid reds. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes red areas more or less intense. Right makes reds more vivid, left more muted. -0.1 to -0.3 for matte reds, 0.1 to 0.3 for vivid ones.",
    },
    "sel_red_light": {
        "short": "Lighter or darker red areas",
        "en": "Luminance shift for red band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1–0.2 lightens skin tones. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes red areas lighter or darker. Right lightens them, left darkens them. 0.1 to 0.2 lightens skin tones nicely.",
    },
    "sel_yellow_hue": {
        "short": "Shifts yellow hue toward green or orange",
        "en": "Shifts hue of yellow band (60°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Yellow sits at 60°: +10 shifts toward green, -10 toward orange. Range -30 to +30 degrees.",
        "what": "Shifts the hue of yellow areas toward green or toward orange. Right moves yellow toward green, left toward orange. 10 toward green, -10 toward orange.",
    },
    "sel_yellow_sat": {
        "short": "More or less yellow intensity",
        "en": "Luminance-preserving saturation for yellow band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vibrant yellows. Range -1 to +1.",
        "what": "Makes yellow areas more or less intense. Right makes yellows more vivid, left more muted. 0.1 to 0.3 for vibrant yellows.",
    },
    "sel_yellow_light": {
        "short": "Lighter or darker yellow areas",
        "en": "Luminance shift for yellow band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1–0.2 lightens skin tones in yellow areas. Range -1 to +1.",
        "what": "Makes yellow areas lighter or darker. Right lightens them, left darkens them. 0.1 to 0.2 lightens skin tones in yellow areas.",
    },
    "sel_green_hue": {
        "short": "Shifts green hue toward cyan or yellow",
        "en": "Shifts hue of green band (120°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "+10 to +20 shifts toward cyan for more natural vegetation. Range -30 to +30.",
        "what": "Shifts the hue of green areas toward cyan or toward yellow. Right moves green toward cyan, left toward yellow. 10 to 20 toward cyan gives more natural vegetation.",
    },
    "sel_green_sat": {
        "short": "More or less green intensity",
        "en": "Luminance-preserving saturation for green band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid greens. Range -1 to +1.",
        "what": "Makes green areas more or less intense. Right makes greens more vivid, left more muted. 0.1 to 0.3 for vivid greens.",
    },
    "sel_green_light": {
        "short": "Lighter or darker green areas",
        "en": "Luminance shift for green band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "+0.1 lightens vegetation. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes green areas lighter or darker. Right lightens them, left darkens them. 0.1 lightens vegetation.",
    },
    "sel_cyan_hue": {
        "short": "Shifts cyan hue toward blue or green",
        "en": "Shifts hue of cyan band (180°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward blue (+10) or green (-10). Range -30 to +30 degrees.",
        "what": "Shifts the hue of cyan areas toward blue or toward green. Right moves cyan toward blue, left toward green. 10 toward blue, -10 toward green.",
    },
    "sel_cyan_sat": {
        "short": "More or less cyan intensity",
        "en": "Luminance-preserving saturation for cyan band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid cyans. Range -1 to +1.",
        "what": "Makes cyan areas more or less intense. Right makes cyans more vivid, left more muted. 0.1 to 0.3 for vivid cyans.",
    },
    "sel_cyan_light": {
        "short": "Lighter or darker cyan areas",
        "en": "Luminance shift for cyan band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust sky brightness in cyan areas. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes cyan areas lighter or darker. Right lightens them, left darkens them. Use it to adjust the brightness of cyan areas like sky.",
    },
    "sel_blue_hue": {
        "short": "Shifts blue hue toward cyan or violet",
        "en": "Shifts hue of blue band (240°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward cyan (-10) or violet (+10). Range -30 to +30 degrees.",
        "what": "Shifts the hue of blue areas toward cyan or toward violet. Right moves blue toward violet, left toward cyan. 10 toward violet, -10 toward cyan.",
    },
    "sel_blue_sat": {
        "short": "More or less blue intensity",
        "en": "Luminance-preserving saturation for blue band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "-0.2 mutes sky for a filmic look. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes blue areas more or less intense. Right makes blues more vivid, left more muted. -0.2 mutes the sky for a filmic look.",
    },
    "sel_blue_light": {
        "short": "Lighter or darker blue areas",
        "en": "Luminance shift for blue band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust sky brightness in blue areas. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes blue areas lighter or darker. Right lightens them, left darkens them. Use it to adjust the brightness of blue areas like sky.",
    },
    "sel_magenta_hue": {
        "short": "Shifts magenta hue toward red or blue",
        "en": "Shifts hue of magenta band (300°±60°): h' = (h + delta) mod 360, raised-cosine window.",
        "tip": "Shift toward red (+10) to protect lips. Range -30 to +30 degrees.",
        "what": "Shifts the hue of magenta areas toward red or toward blue. Right moves magenta toward red, left toward blue. 10 toward red protects lips.",
    },
    "sel_magenta_sat": {
        "short": "More or less magenta intensity",
        "en": "Luminance-preserving saturation for magenta band: out = Y + (1+delta_s)*(x-Y).",
        "tip": "Negative for muted, positive for vivid magentas. Range -1 to +1.",
        "what": "Makes magenta areas more or less intense. Right makes magentas more vivid, left more muted. 0.1 to 0.3 for vivid magentas.",
    },
    "sel_magenta_light": {
        "short": "Lighter or darker magenta areas",
        "en": "Luminance shift for magenta band in HSV: V' = V + delta_L, raised-cosine window ±60°.",
        "tip": "Adjust skin tones in magenta areas. Range -1 to +1; typical ±0.1–0.3.",
        "what": "Makes magenta areas lighter or darker. Right lightens them, left darkens them. Use it to adjust skin tones in magenta areas.",
    },
    # ── SLIDERS: split ─────────────────────────────────────────────────────
    "shadow_hue": {
        "short": "Color direction of the shadow tint",
        "en": "Hue angle for shadow tint in Lab: ab += amount*30*(cos(hue), sin(hue)), weighted by shadow zone.",
        "tip": "30 = warm/orange, 220 = cool/blue. Range 0–360; affects only the shadow zone.",
        "what": "Chooses the color direction of the tint added to the shadow areas. Move it to change the tint from warm orange to cool blue. Only the shadow zone is affected.",
    },
    "shadow_amount": {
        "short": "Strength of the shadow tint",
        "en": "Strength of the Lab a*/b* offset in shadows: ab_offset = amount * 30 * (cos h, sin h).",
        "tip": "0 = no mix, 1 = full. 0.1–0.3 for a subtle effect.",
        "what": "Sets how strong the color tint in the shadow areas is. Right adds more color to the shadows, left less. 0.1 to 0.3 for a subtle split-tone effect.",
    },
    "highlight_hue": {
        "short": "Color direction of the highlight tint",
        "en": "Hue angle for highlight tint in Lab: ab += amount*30*(cos(hue), sin(hue)), weighted by highlight zone.",
        "tip": "40 = warm/golden, 200 = cool/blue. Range 0–360; affects only the highlight zone.",
        "what": "Chooses the color direction of the tint added to the highlight areas. Move it to change the tint from warm golden to cool blue. Only the highlight zone is affected.",
    },
    "highlight_amount": {
        "short": "Strength of the highlight tint",
        "en": "Strength of the Lab a*/b* offset in highlights: ab_offset = amount * 30 * (cos h, sin h).",
        "tip": "0 = no mix, 1 = full. 0.1–0.3 for a subtle effect.",
        "what": "Sets how strong the color tint in the highlight areas is. Right adds more color to the highlights, left less. 0.1 to 0.3 for a subtle split-tone effect.",
    },
    # ── SLIDERS: rgb ───────────────────────────────────────────────────────
    "rgb_m_gamma": {
        "short": "Master midtone brightness (all channels)",
        "en": "Master gamma (global): out = ch^(1/gamma), applied to all channels before per-channel controls.",
        "tip": "1.0 = no change. 0.9 darkens, 1.1 brightens midtones. Range 0.2–5.0.",
        "what": "Brightens or darkens the middle tones of all three color channels. Right brightens the midtones, left darkens them. 1.0 is no change, 0.9 darkens, 1.1 brightens.",
    },
    "rgb_m_pedestal": {
        "short": "Master brightness offset (all channels)",
        "en": "Master offset (global): out = out + offset, added to all channels after gamma and gain.",
        "tip": "Positive lifts blacks, negative darkens. Range -0.5 to 0.5; typical ±0.02–0.1.",
        "what": "Shifts the brightness of every tone equally across all channels. Right lifts the whole image including the blacks, left darkens it. 0.02 to 0.1 is a typical range.",
    },
    "rgb_m_gain": {
        "short": "Master overall brightness (all channels)",
        "en": "Master gain (global): out = out * gain, multiplied on all channels after gamma.",
        "tip": "1.0 = no change. >1 brightens, <1 darkens. Range 0–4.0; typical 0.8–1.3.",
        "what": "Scales the overall brightness of all three channels. Right brightens the whole image, left darkens it. 1.0 is no change, 0.8 to 1.3 is typical.",
    },
    "rgb_r_gamma": {
        "short": "Red midtone brightness (zonal)",
        "en": "Red channel gamma (zonal): out = ch^(1/gamma), blended with zone masks (shadows/mids/highlights).",
        "tip": "1.0 = no change. <1 darkens reds, >1 brightens. Range 0.2–5.0.",
        "what": "Brightens or darkens the red channel in the zones selected below. Above 1.0 the reds get brighter and warmer, below 1.0 they get darker. 1.0 is no change.",
    },
    "rgb_r_pedestal": {
        "short": "Red brightness offset (zonal)",
        "en": "Red channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds red (warm), negative removes it. Range -0.5 to 0.5.",
        "what": "Adds or removes red across the selected zones. Right adds red and warms the image, left removes red. 0.02 to 0.1 is a typical range.",
    },
    "rgb_r_gain": {
        "short": "Red overall brightness (zonal)",
        "en": "Red channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts reds, <1 reduces. Range 0–4.0.",
        "what": "Boosts or reduces the red channel in the selected zones. Above 1.0 the reds get stronger, below 1.0 weaker. 1.0 is no change.",
    },
    "rgb_g_gamma": {
        "short": "Green midtone brightness (zonal)",
        "en": "Green channel gamma (zonal): out = ch^(1/gamma), blended with zone masks.",
        "tip": "1.0 = no change. <1 darkens greens, >1 brightens. Range 0.2–5.0.",
        "what": "Brightens or darkens the green channel in the zones selected below. Above 1.0 the greens get brighter, below 1.0 darker. 1.0 is no change.",
    },
    "rgb_g_pedestal": {
        "short": "Green brightness offset (zonal)",
        "en": "Green channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds green, negative removes it. Range -0.5 to 0.5.",
        "what": "Adds or removes green across the selected zones. Right adds green, left removes it. 0.02 to 0.1 is a typical range.",
    },
    "rgb_g_gain": {
        "short": "Green overall brightness (zonal)",
        "en": "Green channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts greens, <1 reduces. Range 0–4.0.",
        "what": "Boosts or reduces the green channel in the selected zones. Above 1.0 the greens get stronger, below 1.0 weaker. 1.0 is no change.",
    },
    "rgb_b_gamma": {
        "short": "Blue midtone brightness (zonal)",
        "en": "Blue channel gamma (zonal): out = ch^(1/gamma), blended with zone masks.",
        "tip": "1.0 = no change. <1 darkens blues, >1 brightens. Range 0.2–5.0.",
        "what": "Brightens or darkens the blue channel in the zones selected below. Above 1.0 the blues get brighter and cooler, below 1.0 darker. 1.0 is no change.",
    },
    "rgb_b_pedestal": {
        "short": "Blue brightness offset (zonal)",
        "en": "Blue channel offset (zonal): out = out + offset — shifts every tone equally, blended with zone masks.",
        "tip": "Positive adds blue (cool), negative removes it. Range -0.5 to 0.5.",
        "what": "Adds or removes blue across the selected zones. Right adds blue and cools the image, left removes it. 0.02 to 0.1 is a typical range.",
    },
    "rgb_b_gain": {
        "short": "Blue overall brightness (zonal)",
        "en": "Blue channel gain (zonal): out = out * gain, blended with zone masks.",
        "tip": "1.0 = no change. >1 boosts blues, <1 reduces. Range 0–4.0.",
        "what": "Boosts or reduces the blue channel in the selected zones. Above 1.0 the blues get stronger, below 1.0 weaker. 1.0 is no change.",
    },
    "rgb_use_shadows": {
        "short": "Enable R/G/B effect in shadows",
        "en": "Scales the shadow zone mask (smoothstep 0–0.35 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in shadows, 0 = off. Master always acts everywhere.",
        "what": "Sets how much the red, green and blue controls act in the shadow areas. Right applies them fully in the shadows, left reduces or switches them off there. The master controls always act everywhere.",
    },
    "rgb_use_midtones": {
        "short": "Enable R/G/B effect in midtones",
        "en": "Scales the midtone zone mask (smoothstep 0.35–0.65 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in midtones, 0 = off. Master always acts everywhere.",
        "what": "Sets how much the red, green and blue controls act in the midtone areas. Right applies them fully in the midtones, left reduces or switches them off there. The master controls always act everywhere.",
    },
    "rgb_use_highlights": {
        "short": "Enable R/G/B effect in highlights",
        "en": "Scales the highlight zone mask (smoothstep 0.65–1.0 luminance) for R/G/B channel controls.",
        "tip": "1.0 = full effect in highlights, 0 = off. Master always acts everywhere.",
        "what": "Sets how much the red, green and blue controls act in the highlight areas. Right applies them fully in the highlights, left reduces or switches them off there. The master controls always act everywhere.",
    },
    # ── SLIDERS: color replace ─────────────────────────────────────────────
    "cr_0_src_hue": {
        "short": "Source hue (instance 1)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_0_src_sat": {
        "short": "Source saturation (instance 1)",
        "en": "Saturation of the picked source color. Describes the color; it does not scale the effect.",
        "tip": "Set by the pipette or the wheel; use Min sat to include pale, grey or white pixels.",
        "what": "The saturation of the source color, set when you pick it. It only describes the color you picked — it does not limit how strongly pixels are replaced. To decide whether pale, grey and white pixels are included, use Min sat instead."
    },
    "cr_0_src_bright": {
        "short": "Source luminance (instance 1)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_0_tgt_hue": {
        "short": "Target hue (instance 1)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_0_tgt_sat": {
        "short": "Target saturation (instance 1)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_0_tgt_bright": {
        "short": "Target luminance (instance 1)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_0_intensity": {
        "short": "Replace strength (instance 1)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_0_range": {
        "short": "Hue range (instance 1)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_1_src_hue": {
        "short": "Source hue (instance 2)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_1_src_sat": {
        "short": "Source saturation (instance 2)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_1_src_bright": {
        "short": "Source luminance (instance 2)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_1_tgt_hue": {
        "short": "Target hue (instance 2)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_1_tgt_sat": {
        "short": "Target saturation (instance 2)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_1_tgt_bright": {
        "short": "Target luminance (instance 2)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_1_intensity": {
        "short": "Replace strength (instance 2)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_1_range": {
        "short": "Hue range (instance 2)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_2_src_hue": {
        "short": "Source hue (instance 3)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_2_src_sat": {
        "short": "Source saturation (instance 3)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_2_src_bright": {
        "short": "Source luminance (instance 3)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_2_tgt_hue": {
        "short": "Target hue (instance 3)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_2_tgt_sat": {
        "short": "Target saturation (instance 3)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_2_tgt_bright": {
        "short": "Target luminance (instance 3)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_2_intensity": {
        "short": "Replace strength (instance 3)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_2_range": {
        "short": "Hue range (instance 3)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_3_src_hue": {
        "short": "Source hue (instance 4)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_3_src_sat": {
        "short": "Source saturation (instance 4)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_3_src_bright": {
        "short": "Source luminance (instance 4)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_3_tgt_hue": {
        "short": "Target hue (instance 4)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_3_tgt_sat": {
        "short": "Target saturation (instance 4)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_3_tgt_bright": {
        "short": "Target luminance (instance 4)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_3_intensity": {
        "short": "Replace strength (instance 4)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_3_range": {
        "short": "Hue range (instance 4)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_4_src_hue": {
        "short": "Source hue (instance 5)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_4_src_sat": {
        "short": "Source saturation (instance 5)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_4_src_bright": {
        "short": "Source luminance (instance 5)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_4_tgt_hue": {
        "short": "Target hue (instance 5)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_4_tgt_sat": {
        "short": "Target saturation (instance 5)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_4_tgt_bright": {
        "short": "Target luminance (instance 5)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_4_intensity": {
        "short": "Replace strength (instance 5)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_4_range": {
        "short": "Hue range (instance 5)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_5_src_hue": {
        "short": "Source hue (instance 6)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_5_src_sat": {
        "short": "Source saturation (instance 6)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_5_src_bright": {
        "short": "Source luminance (instance 6)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_5_tgt_hue": {
        "short": "Target hue (instance 6)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_5_tgt_sat": {
        "short": "Target saturation (instance 6)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_5_tgt_bright": {
        "short": "Target luminance (instance 6)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_5_intensity": {
        "short": "Replace strength (instance 6)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_5_range": {
        "short": "Hue range (instance 6)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_6_src_hue": {
        "short": "Source hue (instance 7)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_6_src_sat": {
        "short": "Source saturation (instance 7)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_6_src_bright": {
        "short": "Source luminance (instance 7)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_6_tgt_hue": {
        "short": "Target hue (instance 7)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_6_tgt_sat": {
        "short": "Target saturation (instance 7)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_6_tgt_bright": {
        "short": "Target luminance (instance 7)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_6_intensity": {
        "short": "Replace strength (instance 7)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_6_range": {
        "short": "Hue range (instance 7)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_7_src_hue": {
        "short": "Source hue (instance 8)",
        "en": "Center hue of the source range in degrees (0–360). Pixels within ±range of this hue are affected.",
        "tip": "Pick the dominant hue you want to replace. Use the color wheel.",
        "what": "Chooses which color in the image this instance works on. Move it to the hue you want to replace, like a blue sky or green grass. The source wheel next to it shows the selected color."
    },
    "cr_7_src_sat": {
        "short": "Source saturation (instance 8)",
        "en": "Scales the mask: higher = only saturated pixels affected. 0 = no effect, 1 = full.",
        "tip": "1.0 for vivid colors, 0.5 for pastel tones.",
        "what": "Sets how saturated the source color has to be for this instance to act. Right only affects vivid colors, left also includes pale and greyish ones. 1.0 for vivid colors, 0.5 for pastel tones."
    },
    "cr_7_src_bright": {
        "short": "Source luminance (instance 8)",
        "en": "Shifts the luminance match window. 0 = mid, -1 = shadows only, 1 = highlights only.",
        "tip": "0 matches all brightness, -0.5 favors shadows, 0.5 favors highlights.",
        "what": "Sets which brightness range the source color has to have. Right only affects bright areas, left only dark ones. 0 matches all brightness levels."
    },
    "cr_7_tgt_hue": {
        "short": "Target hue (instance 8)",
        "en": "Replacement hue in degrees (0–360). Affected pixels blend toward this hue.",
        "tip": "Pick the desired replacement color on the target wheel.",
        "what": "Chooses the replacement color for the matched areas. Move it to the hue you want the source color to turn into. The target wheel next to it shows the selected replacement color."
    },
    "cr_7_tgt_sat": {
        "short": "Target saturation (instance 8)",
        "en": "Saturation of the replacement color. 0 = grey, 1 = fully saturated.",
        "tip": "Match the source saturation or adjust for a muted look.",
        "what": "Sets how saturated the replacement color is. Right makes the replacement more vivid, left more muted. Match the source saturation or lower it for a muted look."
    },
    "cr_7_tgt_bright": {
        "short": "Target luminance (instance 8)",
        "en": "Adds brightness offset to the replacement color. -1 = darker, 0 = unchanged, 1 = brighter.",
        "tip": "0 keeps original brightness, -0.3 darkens, 0.3 lightens replaced pixels.",
        "what": "Makes the replacement color brighter or darker than the original. Right lightens the replaced areas, left darkens them. 0 keeps the original brightness."
    },
    "cr_7_intensity": {
        "short": "Replace strength (instance 8)",
        "en": "Blend factor: out = x + mask * intensity * (target - x). 0 = off, 1 = full replacement.",
        "tip": "0.5 for subtle shifts, 1.0 for complete replacement.",
        "what": "Sets how strongly this instance replaces the source color. Right replaces it more completely, left blends it in more subtly. 0.5 for subtle shifts, 1.0 for a complete replacement."
    },
    "cr_7_range": {
        "short": "Hue range (instance 8)",
        "en": "Half-width of the hue window in degrees. mask = max(0, 1 - |Δhue|/range).",
        "tip": "30° for a narrow band, 90°+ for a wide range.",
        "what": "Sets how wide the hue range is around the source color. Right includes more neighboring colors, left only a narrow band. 30 degrees for a narrow band, 90 or more for a wide range."
    },
    "cr_0_sat_floor": {
        "short": "Min saturation (instance 1)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_1_sat_floor": {
        "short": "Min saturation (instance 2)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_2_sat_floor": {
        "short": "Min saturation (instance 3)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_3_sat_floor": {
        "short": "Min saturation (instance 4)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_4_sat_floor": {
        "short": "Min saturation (instance 5)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_5_sat_floor": {
        "short": "Min saturation (instance 6)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_6_sat_floor": {
        "short": "Min saturation (instance 7)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
    },
    "cr_7_sat_floor": {
        "short": "Min saturation (instance 8)",
        "en": "Saturation at which the effect is fully on. Lower = also affects pale and grey pixels. 0 = no guard (affects white).",
        "tip": "0.15 default, lower to reach pale colors, 0 to affect white and grey.",
        "what": "Sets how saturated a pixel must be for this instance to act on it. Right only affects vivid colors, left also reaches pale, grey and white ones. 0.15 is the default; 0 turns the guard off entirely so even white and grey pixels are replaced."
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

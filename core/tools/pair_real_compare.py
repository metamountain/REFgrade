"""Real pairs (bull + FiveK subset): fit on the left half, measure dE76 on the right half.
Compares the strategies from pair_strategies on GPU. Writes reports/pair_real_compare.md."""
import sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from refgrade.color import rgb_to_lab
from refgrade.lut import apply as lut_apply
from refgrade.pairlut import as_lut
import pair_strategies as PS

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
NAMES = ["ls", "splat", "splat_blur0.7", "splat_blur1.0"]

def load(p):
    im = Image.open(p).convert("RGB"); im.thumbnail((768, 768))
    return torch.from_numpy(np.asarray(im, np.float32) / 255)[None].to(DEV)

pairs = [(ROOT / "Dual LUT" / "bull1.jpg", ROOT / "Dual LUT" / "Bull2.jpg")]
pairs += [(b, Path(str(b).replace("_before", "_after"))) for b in sorted((ROOT / "research" / "pairs_fivek").glob("*_before.png"))]
res = {n: [] for n in NAMES}; ms = {n: [] for n in NAMES}
for i, (bp, ap) in enumerate(pairs):
    x, y = load(bp), load(ap); w = x.shape[2] // 2
    xl, yl, xr, yr = x[:, :, :w].contiguous(), y[:, :, :w].contiguous(), x[:, :, w:].contiguous(), y[:, :, w:].contiguous()
    for n in NAMES:
        t = time.perf_counter(); tab = PS.STRATEGIES[n](xl.reshape(-1, 3), yl.reshape(-1, 3))
        if DEV.type == "cuda": torch.cuda.synchronize()
        ms[n].append(1000 * (time.perf_counter() - t))
        out = lut_apply(xr, as_lut(torch.as_tensor(tab).to(DEV).permute(2, 1, 0, 3).float())).clamp(0, 1)
        res[n].append((rgb_to_lab(out) - rgb_to_lab(yr)).norm(dim=-1).mean().item())
    print(i, " ".join(f"{n}={res[n][-1]:.2f}" for n in NAMES), flush=True)
with open(ROOT / "reports" / "pair_real_compare.md", "w", encoding="utf-8") as f:
    f.write(f"# Real pairs, held out (left fit -> right measured), {len(pairs)} pairs, {DEV}\n\n| strategy | bull | FiveK median | FiveK mean | wins | ms median |\n|---|---|---|---|---|---|\n")
    arr = np.array([res[n] for n in NAMES]); wins = np.bincount(arr[:, 1:].argmin(0), minlength=len(NAMES))
    for k, n in enumerate(NAMES):
        f.write(f"| {n} | {res[n][0]:.2f} | {np.median(res[n][1:]):.2f} | {np.mean(res[n][1:]):.2f} | {wins[k]} | {np.median(ms[n]):.0f} |\n")
print(open(ROOT / "reports" / "pair_real_compare.md", encoding="utf-8").read())

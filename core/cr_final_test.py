"""Final CR test: fresh worker, red barn -> blue. Usage: cr_final_test.py <image>"""
import subprocess, sys, time, urllib.request, json, socket, tempfile
import numpy as np
from pathlib import Path
from PIL import Image

if len(sys.argv) < 2:
    print("usage: cr_final_test.py <source-image>")
    sys.exit(2)

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]

tmp = Path(tempfile.gettempdir()) / f"cr_final_{port}"
tmp.mkdir(exist_ok=True)
proc = subprocess.Popen(
    [sys.executable, "-E", "-m", "worker", "--port", str(port), "--tmp", str(tmp)],
    cwd=str(CORE), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for _ in range(60):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        break
    except OSError:
        time.sleep(0.5)
else:
    print("FAIL: worker not up")
    proc.kill()
    sys.exit(1)

print(f"Worker on port {port}")

def call(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

src = sys.argv[1]
a = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
h, w = a.shape[:2]
raw = tmp / "src.raw"
np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)]).tofile(raw)
r = call("/load", {"raw": str(raw), "width": w, "height": h, "depth": "U8"})
sid = r["session"]
print(f"Loaded: {sid} {w}x{h}")

mix = {"weights": {k: 1.0 for k in ("reference", "color", "tone", "rgb",
                                    "curves", "wheels", "hsl", "color_replace", "lut")},
       "master": 1.0, "solo": None}
mix["weights"]["reference"] = 0.0

vals_cr = {"cr_0_src_hue": 0, "cr_0_src_sat": 1, "cr_0_tgt_hue": 240,
           "cr_0_tgt_sat": 1, "cr_0_intensity": 1, "cr_0_range": 30}

out = tmp / "cr_blue.raw"
call("/apply", {"session": sid, "references": [src], "method": "reinhard",
                "values": vals_cr, "out": str(out), "mix": mix})
bgra = np.fromfile(out, dtype=np.uint8).reshape(h, w, 4)
rgb = np.ascontiguousarray(bgra[..., [2, 1, 0]])
out_png = tmp / "cr_red_to_blue.png"
Image.fromarray(rgb).save(out_png)
print(f"CR result: {out_png}")

out2 = tmp / "no_cr.raw"
call("/apply", {"session": sid, "references": [src], "method": "reinhard",
                "values": {}, "out": str(out2), "mix": mix})
bgra2 = np.fromfile(out2, dtype=np.uint8).reshape(h, w, 4)
rgb2 = np.ascontiguousarray(bgra2[..., [2, 1, 0]])
diff = np.abs(rgb.astype(int) - rgb2.astype(int)).mean()
print(f"Pixel diff: {diff:.2f}")
print(f"CR WORKS: {'YES' if diff > 1 else 'NO'}")

# Copy to project for viewing
dest = CORE / "cr_red_to_blue.png"
Image.fromarray(rgb).save(dest)
print(f"Copied to: {dest}")

proc.terminate()
proc.wait(5)

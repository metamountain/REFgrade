"""ONNX LUT application - prototype für eine GPU-native Pipeline.

Zwei Modelle:
- ``create_identity_model``: reiner Smoke-Test (Identity), misst nur den
  ONNX-Runtime-Overhead.
- ``create_lut_apply_model``: echte trilineare 3D-LUT-Interpolation,
  äquivalent zu ``refgrade.lut.apply`` (grid_sample, align_corners=True).

Konvention:
  Input  image [1, 3, H, W]    float32, 0..1
         lut   [1, 3, N, N, N] float32, 0..1, Raumachsen (D=b, H=g, W=r)
  Output result [1, 3, H, W]   float32, 0..1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Identity (Smoke-Test)
# ---------------------------------------------------------------------------
def create_identity_model():
    """ONNX-Modell: result = image (reiner Overhead-Smoke-Test)."""
    from onnx import TensorProto, helper

    nodes = [helper.make_node("Identity", ["image"], ["result"])]
    image_in = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"])
    result_out = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 3, "H", "W"])
    graph = helper.make_graph(nodes, "lut_identity", [image_in], [result_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# Echte trilineare LUT-Interpolation
# ---------------------------------------------------------------------------
def create_lut_apply_model(lut_size: int = 33):
    """ONNX-Modell für trilineare 3D-LUT-Interpolation.

    Nachbildung von ``refgrade.lut.apply``:
      pos = pixel * (N-1);  trilineares Blending der 8 Eckwerte.
    """
    from onnx import TensorProto, helper, numpy_helper

    n = int(lut_size)
    nodes: list = []
    consts: dict[str, np.ndarray] = {}

    def const(name: str, arr: np.ndarray) -> None:
        consts[name] = arr

    # --- Konstanten -------------------------------------------------------
    const("scale", np.array(float(n - 1), dtype=np.float32))
    const("one_f", np.array(1.0, dtype=np.float32))
    const("zero_f", np.array(0.0, dtype=np.float32))
    const("clip_max", np.array(float(n - 1), dtype=np.float32))
    const("c_idx", np.arange(3, dtype=np.int64).reshape(1, 1, 1, 3, 1))
    const("i0", np.array(0, dtype=np.int64))
    const("i1", np.array(1, dtype=np.int64))
    # 1-D-Varianten für Shape-Gather (→ [1]) und Shape-Konstruktion (Concat axis=0)
    const("s_i1", np.array([1], dtype=np.int64))
    const("s_i2", np.array([2], dtype=np.int64))
    const("s_i3", np.array([3], dtype=np.int64))
    const("s0", np.array([0], dtype=np.int64))
    const("s_i8", np.array([8], dtype=np.int64))
    const("sl_starts0", np.array([0], dtype=np.int64))
    const("sl_ends0", np.array([1], dtype=np.int64))
    const("sl_starts1", np.array([1], dtype=np.int64))
    const("sl_ends1", np.array([2], dtype=np.int64))
    const("sl_starts2", np.array([2], dtype=np.int64))
    const("sl_ends2", np.array([3], dtype=np.int64))
    const("sl_axes2", np.array([2], dtype=np.int64))

    # --- image [1,3,H,W] -> [1,H,W,3] -> [1,P,3]  (P = H*W, dynamisch) ----
    nodes.append(helper.make_node("Shape", ["image"], ["shape"]))
    nodes.append(helper.make_node("Gather", ["shape", "s_i2"], ["H"], axis=0))
    nodes.append(helper.make_node("Gather", ["shape", "s_i3"], ["W"], axis=0))
    nodes.append(helper.make_node("Mul", ["H", "W"], ["P"]))
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i3"], ["reshape_pts"], axis=0))
    nodes.append(helper.make_node("Transpose", ["image"], ["img_hwc"], perm=[0, 2, 3, 1]))
    nodes.append(helper.make_node("Reshape", ["img_hwc", "reshape_pts"], ["pts"]))

    # --- pos = pts * (N-1), dann r/g/b splitten ([1,P,1]) -----------------
    nodes.append(helper.make_node("Mul", ["pts", "scale"], ["pos"]))
    nodes.append(helper.make_node("Slice", ["pos", "sl_starts0", "sl_ends0", "sl_axes2"], ["r_pos"]))
    nodes.append(helper.make_node("Slice", ["pos", "sl_starts1", "sl_ends1", "sl_axes2"], ["g_pos"]))
    nodes.append(helper.make_node("Slice", ["pos", "sl_starts2", "sl_ends2", "sl_axes2"], ["b_pos"]))

    for nm in ("r", "g", "b"):
        nodes.append(helper.make_node("Floor", [f"{nm}_pos"], [f"{nm}0f"]))
        nodes.append(helper.make_node("Add", [f"{nm}0f", "one_f"], [f"{nm}1f"]))
        nodes.append(helper.make_node("Sub", [f"{nm}_pos", f"{nm}0f"], [f"{nm}w"]))
        nodes.append(helper.make_node("Clip", [f"{nm}0f", "zero_f", "clip_max"], [f"{nm}0"]))
        nodes.append(helper.make_node("Clip", [f"{nm}1f", "zero_f", "clip_max"], [f"{nm}1"]))

    # --- 8 Eck-Indizes (b,g,r) -> [1,P,8,3] -> [1,P,8,3,1] ----------------
    corners = [
        ("r0", "g0", "b0"), ("r1", "g0", "b0"),
        ("r0", "g1", "b0"), ("r1", "g1", "b0"),
        ("r0", "g0", "b1"), ("r1", "g0", "b1"),
        ("r0", "g1", "b1"), ("r1", "g1", "b1"),
    ]
    for i, (rr, gg, bb) in enumerate(corners):
        nodes.append(helper.make_node("Concat", [bb, gg, rr], [f"corner{i}"], axis=-1))
    nodes.append(helper.make_node("Concat", [f"corner{i}" for i in range(8)], ["corner_stack"], axis=2))
    nodes.append(helper.make_node("Cast", ["corner_stack"], ["corner_stack_int"], to=TensorProto.INT64))

    # --- LUT [1,3,N,N,N] -> [3,N,N,N] (c,b,g,r) --------------------------
    nodes.append(helper.make_node("Squeeze", ["lut", "s0"], ["lut4d"]))

    # --- GatherND-Indizes [1,P,8,3,4] = (c, b, g, r) ---------------------
    # A (Raum): corner_stack_int [1,P,8,3] -> [1,P,8,1,3] -> Expand [1,P,8,3,3]
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i8", "s_i1", "s_i3"], ["spatial_5"], axis=0))
    nodes.append(helper.make_node("Reshape", ["corner_stack_int", "spatial_5"], ["spatial_5r"]))  # [1,P,8,1,3]
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i8", "s_i3", "s_i3"], ["spatial_exp"], axis=0))
    nodes.append(helper.make_node("Expand", ["spatial_5r", "spatial_exp"], ["idx_spatial"]))  # [1,P,8,3,3] = (b,g,r)
    # B (Kanal): c_idx [1,1,1,3,1] -> Expand [1,P,8,3,1]
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i8", "s_i3", "s_i1"], ["chan_exp"], axis=0))
    nodes.append(helper.make_node("Expand", ["c_idx", "chan_exp"], ["idx_chan"]))  # [1,P,8,3,1] = c
    # Concat: (c, b, g, r)
    nodes.append(helper.make_node("Concat", ["idx_chan", "idx_spatial"], ["gather_idx"], axis=-1))  # [1,P,8,3,4]
    nodes.append(helper.make_node("GatherND", ["lut4d", "gather_idx"], ["corners8"]))  # [1,P,8,3]

    # --- [1,P,8,3] -> [1,P,2,2,2,3]  (Achse 8 = b*4 + g*2 + r) -----------
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i2", "s_i2", "s_i2", "s_i3"], ["reshape_8"], axis=0))
    nodes.append(helper.make_node("Reshape", ["corners8", "reshape_8"], ["corners8r"]))

    # --- Gewichte [1,P,1] -> rank-angepasstes [1,P,1,...,1] --------------
    # r: [1,P,1,1,1]  g: [1,P,1,1]  b: [1,P,1]
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i1", "s_i1", "s_i1"], ["rw_shape"], axis=0))
    nodes.append(helper.make_node("Reshape", ["rw", "rw_shape"], ["rw5"]))
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i1", "s_i1"], ["gw_shape"], axis=0))
    nodes.append(helper.make_node("Reshape", ["gw", "gw_shape"], ["gw5"]))
    nodes.append(helper.make_node("Concat", ["s_i1", "P", "s_i1"], ["bw_shape"], axis=0))
    nodes.append(helper.make_node("Reshape", ["bw", "bw_shape"], ["bw5"]))
    nodes.append(helper.make_node("Sub", ["one_f", "rw5"], ["rw1"]))
    nodes.append(helper.make_node("Sub", ["one_f", "gw5"], ["gw1"]))
    nodes.append(helper.make_node("Sub", ["one_f", "bw5"], ["bw1"]))

    # --- trilineares Blending --------------------------------------------
    # lerp entlang r (Achse 4) -> [1,P,2,2,3]
    nodes.append(helper.make_node("Gather", ["corners8r", "i0"], ["c_r0"], axis=4))
    nodes.append(helper.make_node("Gather", ["corners8r", "i1"], ["c_r1"], axis=4))
    nodes.append(helper.make_node("Mul", ["c_r0", "rw1"], ["t_r0"]))
    nodes.append(helper.make_node("Mul", ["c_r1", "rw5"], ["t_r1"]))
    nodes.append(helper.make_node("Add", ["t_r0", "t_r1"], ["lerp_r"]))
    # lerp entlang g (Achse 3) -> [1,P,2,3]
    nodes.append(helper.make_node("Gather", ["lerp_r", "i0"], ["c_g0"], axis=3))
    nodes.append(helper.make_node("Gather", ["lerp_r", "i1"], ["c_g1"], axis=3))
    nodes.append(helper.make_node("Mul", ["c_g0", "gw1"], ["t_g0"]))
    nodes.append(helper.make_node("Mul", ["c_g1", "gw5"], ["t_g1"]))
    nodes.append(helper.make_node("Add", ["t_g0", "t_g1"], ["lerp_g"]))
    # lerp entlang b (Achse 2) -> [1,P,3]
    nodes.append(helper.make_node("Gather", ["lerp_g", "i0"], ["c_b0"], axis=2))
    nodes.append(helper.make_node("Gather", ["lerp_g", "i1"], ["c_b1"], axis=2))
    nodes.append(helper.make_node("Mul", ["c_b0", "bw1"], ["t_b0"]))
    nodes.append(helper.make_node("Mul", ["c_b1", "bw5"], ["t_b1"]))
    nodes.append(helper.make_node("Add", ["t_b0", "t_b1"], ["out_flat"]))

    # --- [1,P,3] -> [1,H,W,3] -> [1,3,H,W] --------------------------------
    nodes.append(helper.make_node("Concat", ["s_i1", "H", "W", "s_i3"], ["reshape_hwc"], axis=0))
    nodes.append(helper.make_node("Reshape", ["out_flat", "reshape_hwc"], ["out_hwc"]))
    nodes.append(helper.make_node("Transpose", ["out_hwc"], ["result"], perm=[0, 3, 1, 2]))

    # --- Modell verpacken -------------------------------------------------
    for name, arr in consts.items():
        nodes.insert(0, helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr)))

    image_in = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"])
    lut_in = helper.make_tensor_value_info("lut", TensorProto.FLOAT, [1, 3, n, n, n])
    result_out = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 3, "H", "W"])
    graph = helper.make_graph(nodes, "lut_apply", [image_in, lut_in], [result_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


# ---------------------------------------------------------------------------
# Speichern / Inferenz
# ---------------------------------------------------------------------------
def save_lut_model(path: str = "core/refgrade/models/lut_apply.onnx", lut_size: int = 33) -> str:
    """Echtes LUT-Modell speichern und validieren."""
    import onnx

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model = create_lut_apply_model(lut_size)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[ONNX] LUT-Modell gespeichert: {path} (N={lut_size})")
    return path


def save_identity_model(path: str = "core/refgrade/models/lut_identity.onnx") -> str:
    """Identity-Smoke-Test-Modell speichern und validieren."""
    import onnx

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model = create_identity_model()
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[ONNX] Identity-Modell gespeichert: {path}")
    return path


def _session(model_path: str, provider: str):
    import onnxruntime as ort

    return ort.InferenceSession(model_path, providers=[provider])


def run_onnx_inference(image_np, lut_np, model_path: str = "core/refgrade/models/lut_apply.onnx",
                       provider: str = "CPUExecutionProvider"):
    """Echte LUT-Inferenz. image [1,3,H,W], lut [1,3,N,N,N] -> [1,3,H,W]."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(model_path).exists():
        save_lut_model(model_path)
    sess = _session(model_path, provider)
    feeds = {i.name: v for i, v in zip(sess.get_inputs(), [image_np, lut_np])}
    return sess.run(None, feeds)[0]


def run_identity_inference(image_np, model_path: str = "core/refgrade/models/lut_identity.onnx",
                           provider: str = "CPUExecutionProvider"):
    """Identity-Inferenz (Overhead-Messung). image [1,3,H,W] -> [1,3,H,W]."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(model_path).exists():
        save_identity_model(model_path)
    sess = _session(model_path, provider)
    return sess.run(None, {sess.get_inputs()[0].name: image_np})[0]


# ---------------------------------------------------------------------------
# Torch-Referenz (Benchmark + Korrektheit)
# ---------------------------------------------------------------------------
def torch_lut_apply(image_np, lut_np):
    """Torch-Referenz, exakt wie ``refgrade.lut.apply``.

    image_np [1,3,H,W] 0..1, lut_np [1,3,N,N,N] (c,b,g,r) -> [1,3,H,W].
    """
    import torch

    from .lut import apply

    img = torch.from_numpy(image_np).permute(0, 2, 3, 1).contiguous()  # [1,H,W,3]
    table = torch.from_numpy(lut_np).squeeze(0).permute(1, 2, 3, 0).contiguous()  # [N,N,N,3] (b,g,r,c)
    lut = {"size": int(lut_np.shape[-1]), "table": table, "title": "onnx-bench"}
    out = apply(img, lut)  # [1,H,W,3]
    return out.permute(0, 3, 1, 2).contiguous().numpy()  # [1,3,H,W]


def make_identity_lut(lut_size: int = 33):
    """Identitäts-LUT [1,3,N,N,N] (c,b,g,r): lut[c,b,g,r] = (r,g,b)."""
    n = int(lut_size)
    r = np.linspace(0.0, 1.0, n).astype(np.float32)
    g = np.linspace(0.0, 1.0, n).astype(np.float32)
    b = np.linspace(0.0, 1.0, n).astype(np.float32)
    rr = np.broadcast_to(r[None, None, :], (n, n, n))  # [n,n,n] = r
    gg = np.broadcast_to(g[None, :, None], (n, n, n))  # [n,n,n] = g
    bb = np.broadcast_to(b[:, None, None], (n, n, n))  # [n,n,n] = b
    table = np.stack([rr, gg, bb], axis=0)  # [3,n,n,n], Index [c][b][g][r]
    return table[np.newaxis].copy()  # [1,3,n,n,n]


def make_test_lut(lut_size: int = 33, seed: int = 0):
    """Nicht-triviale Test-LUT [1,3,N,N,N]: glatte, monotone Farbverschiebung."""
    rng = np.random.default_rng(seed)
    n = int(lut_size)
    base = make_identity_lut(n)[0]  # [3,n,n,n] (c,b,g,r)
    # Glatter, glättender Bias pro Kanal (simuliert einen echten Grade)
    bias = rng.uniform(-0.05, 0.05, size=(3, 1, 1, 1)).astype(np.float32)
    gain = rng.uniform(0.9, 1.1, size=(3, 1, 1, 1)).astype(np.float32)
    table = (base * gain + bias).clip(0.0, 1.0)
    return table[np.newaxis]  # [1,3,n,n,n]

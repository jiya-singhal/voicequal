"""Train the distilled P.835 quality model and export weights (npz + ONNX).

Reads benchmarks/distill/data/dataset.npz (from label.py). Trains a small
MLP on the 169-d clip features to predict DNSMOS (sig, bak, ovrl), using
the train split (28 speakers) and evaluating on the test split (2 unseen
speakers). Also reports the DSP-only baseline (a linear fit from
snr_estimate) so the gain from the mel features is visible.

Writes:
  src/voicequal/models/quality_mlp.npz   numpy inference weights (shipped)
  benchmarks/distill/quality_mlp.onnx    ONNX export of the same network
  benchmarks/distill/results/train_v<version>.json

Usage:
    pip install scikit-learn onnx
    python benchmarks/distill/train.py [--hidden 64 32] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from voicequal import __version__
from voicequal.features import FEATURE_NAMES, N_FEATURES
from voicequal.models.quality import QualityModel

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dataset.npz"
WEIGHTS = HERE.parent.parent / "src" / "voicequal" / "models" / "quality_mlp.npz"
ONNX_OUT = HERE / "quality_mlp.onnx"
RESULTS = HERE / "results"
TARGETS = ("sig", "bak", "ovrl")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(TARGETS):
        out[name] = {
            "pearson": float(pearsonr(y_true[:, j], y_pred[:, j])[0]),
            "spearman": float(spearmanr(y_true[:, j], y_pred[:, j]).correlation),
            "mae": float(np.mean(np.abs(y_true[:, j] - y_pred[:, j]))),
        }
    return out


def export_onnx(scaler: StandardScaler, mlp: MLPRegressor, path: Path) -> int:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    inits = [
        numpy_helper.from_array(scaler.mean_.astype(np.float32), "mean"),
        numpy_helper.from_array(scaler.scale_.astype(np.float32), "scale"),
    ]
    nodes.append(helper.make_node("Sub", ["features", "mean"], ["centered"]))
    nodes.append(helper.make_node("Div", ["centered", "scale"], ["h0"]))
    prev = "h0"
    n = len(mlp.coefs_)
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_, strict=True)):
        inits.append(numpy_helper.from_array(w.astype(np.float32), f"W{i}"))
        inits.append(numpy_helper.from_array(b.astype(np.float32), f"b{i}"))
        nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [f"mm{i}"]))
        out = "raw" if i == n - 1 else f"a{i}"
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [out]))
        if i < n - 1:
            nodes.append(helper.make_node("Relu", [out], [f"h{i + 1}"]))
            prev = f"h{i + 1}"
    inits.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), "mos_min"))
    inits.append(numpy_helper.from_array(np.array(5.0, dtype=np.float32), "mos_max"))
    nodes.append(helper.make_node("Clip", ["raw", "mos_min", "mos_max"], ["mos"]))
    graph = helper.make_graph(
        nodes,
        "voicequal_quality_mlp",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", N_FEATURES])],
        [helper.make_tensor_value_info("mos", TensorProto.FLOAT, ["N", 3])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, producer_name="voicequal", opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=3.0, help="L2 penalty")
    args = ap.parse_args()

    z = np.load(DATA, allow_pickle=False)
    X, Y, split, kind = z["X"], z["Y"], z["split"], z["kind"]
    true_snr = z["true_snr"]
    assert list(z["feature_names"]) == list(FEATURE_NAMES), (
        "feature layout changed; re-run label.py"
    )
    tr, te = split == "train", split == "test"
    print(f"train {tr.sum()} clips ({(kind[tr] == 'clean').sum()} clean), test {te.sum()} clips")

    scaler = StandardScaler().fit(X[tr])
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden),
        alpha=args.alpha,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    mlp.fit(scaler.transform(X[tr]), Y[tr])
    fit_s = time.perf_counter() - t0
    pred_te = np.clip(mlp.predict(scaler.transform(X[te])), 1.0, 5.0)
    pred_tr = np.clip(mlp.predict(scaler.transform(X[tr])), 1.0, 5.0)
    m_te, m_tr = metrics(Y[te], pred_te), metrics(Y[tr], pred_tr)

    # DSP-only baseline: linear map from snr_estimate (feature 0) fitted on train.
    base = {}
    for j, name in enumerate(TARGETS):
        a, b = np.polyfit(X[tr, 0], Y[tr, j], 1)
        pb = np.clip(a * X[te, 0] + b, 1.0, 5.0)
        base[name] = {
            "pearson": float(pearsonr(Y[te, j], pb)[0]),
            "spearman": float(spearmanr(Y[te, j], pb).correlation),
            "mae": float(np.mean(np.abs(Y[te, j] - pb))),
        }

    # Against true SNR on noisy test clips.
    noisy_te = te & (kind == "noisy")
    vs_snr = {
        "student_ovrl": float(spearmanr(pred_te[noisy_te[te], 2], true_snr[noisy_te]).correlation),
        "teacher_ovrl": float(spearmanr(Y[noisy_te, 2], true_snr[noisy_te]).correlation),
        "snr_estimate": float(spearmanr(X[noisy_te, 0], true_snr[noisy_te]).correlation),
    }

    # Save numpy weights.
    n_params = sum(w.size + b.size for w, b in zip(mlp.coefs_, mlp.intercepts_, strict=True))
    save = {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "n_layers": np.array(len(mlp.coefs_)),
        "teacher": np.array("DNSMOS sig_bak_ovr.onnx (microsoft/DNS-Challenge)"),
        "trained_on": np.array(f"VoiceBank-DEMAND train shard 0, {int(tr.sum())} clips"),
        "version": np.array(__version__),
    }
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_, strict=True)):
        save[f"W{i}"] = w.astype(np.float32)
        save[f"b{i}"] = b.astype(np.float32)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WEIGHTS, **save)
    npz_bytes = WEIGHTS.stat().st_size

    # Round-trip check through the shipped numpy inference.
    qm = QualityModel(WEIGHTS)
    rt = np.array([[*qm.predict_features(x).__dict__.values()] for x in X[te][:50]])
    rt_err = float(np.max(np.abs(rt - pred_te[:50])))

    onnx_bytes = export_onnx(scaler, mlp, ONNX_OUT)
    # Verify ONNX matches numpy if onnxruntime is present.
    onnx_err = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"features": X[te][:50].astype(np.float32)})[0]
        onnx_err = float(np.max(np.abs(o - rt)))
    except ImportError:
        pass

    # Inference latency of the numpy model on features.
    t0 = time.perf_counter()
    for x in X[te][:200]:
        qm.predict_features(x)
    infer_us = (time.perf_counter() - t0) / 200 * 1e6

    report = {
        "voicequal_version": __version__,
        "hidden": args.hidden,
        "n_params": int(n_params),
        "npz_bytes": npz_bytes,
        "onnx_bytes": onnx_bytes,
        "fit_seconds": fit_s,
        "train_clips": int(tr.sum()),
        "test_clips": int(te.sum()),
        "test": m_te,
        "train": m_tr,
        "dsp_linear_baseline_test": base,
        "vs_true_snr_noisy_test_spearman": vs_snr,
        "numpy_roundtrip_max_err": rt_err,
        "onnx_vs_numpy_max_err": onnx_err,
        "numpy_inference_us_per_clip": infer_us,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"train_v{__version__}.json").write_text(json.dumps(report, indent=1))

    print(
        f"\nMLP {N_FEATURES}->{'->'.join(map(str, args.hidden))}->3, {n_params} params, "
        f"npz {npz_bytes / 1024:.0f} KB, onnx {onnx_bytes / 1024:.0f} KB, fit {fit_s:.0f}s"
    )
    print(
        f"{'target':<6}{'test r':>8}{'test rho':>10}{'test MAE':>10} | {'baseline r':>11}{'base MAE':>10} | {'train r':>8}"
    )
    for name in TARGETS:
        print(
            f"{name:<6}{m_te[name]['pearson']:>8.3f}{m_te[name]['spearman']:>10.3f}{m_te[name]['mae']:>10.3f} | "
            f"{base[name]['pearson']:>11.3f}{base[name]['mae']:>10.3f} | {m_tr[name]['pearson']:>8.3f}"
        )
    print(
        f"vs true SNR (noisy test, Spearman): student OVRL {vs_snr['student_ovrl']:+.3f}, "
        f"teacher OVRL {vs_snr['teacher_ovrl']:+.3f}, snr_estimate {vs_snr['snr_estimate']:+.3f}"
    )
    print(
        f"numpy round-trip max err {rt_err:.2e}; onnx vs numpy {onnx_err}; inference {infer_us:.0f} us/clip"
    )


if __name__ == "__main__":
    main()

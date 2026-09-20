#!/usr/bin/env python3
"""Train and evaluate the frozen GNN baselines and edge-aware model.

Converted from `07_GNN_Training_and_Baselines.ipynb`. Notebook prose and cell output were intentionally omitted.
"""
# ============================================================
# MODULE 0 — IMPORTS + FROZEN TRAINING CONFIGURATION
# ============================================================
import os, sys, json, time, math, random, copy, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

SEED = 2026
EXPECTED_N = 110
EXPECTED_SPLITS = {"train": 70, "validation": 19, "test": 21}
EXPECTED_DIMS = {"node": 8, "edge": 9, "graph": 12, "target": 2}
TARGET_NAMES = ["von_mises_stress", "equivalent_strain"]

# Conservative, reproducible defaults; validation early stopping avoids wasted GPU time.
HIDDEN_DIM = 128
NUM_LAYERS = 4
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 35
MIN_DELTA = 1e-4

# One graph per optimizer step avoids complicated variable-size batching and
# makes global graph attributes unambiguous. Gradient accumulation gives an
# effective batch of several graphs without large memory spikes.
GRAD_ACCUM_STEPS = 4

# Tiny-overfit gate before full training.
TINY_GRAPHS = 2
TINY_MAX_STEPS = 250
TINY_REQUIRED_FRACTION = 0.35  # final tiny loss must be <35% of initial loss

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

seed_everything()
print("Notebook 07 — GNN Training + Baselines")
print("Frozen split:", EXPECTED_SPLITS)
print("Models: NodeMLP, GraphSAGE, EdgeAwareGNN")

# ============================================================
# MODULE 1 — DRIVE + NOTEBOOK-06 HANDOFF + GPU PREFLIGHT
# ============================================================
PROJECT_ROOT = Path(
    os.environ.get("TPMS_PROJECT_ROOT", Path.cwd() / "TPMS_IEEE_BIGDATA")
).expanduser().resolve()
N06_ROOT = PROJECT_ROOT / "Notebook06"
OUT_ROOT = PROJECT_ROOT / "Notebook07"
CKPT_ROOT = OUT_ROOT / "checkpoints"
PRED_ROOT = OUT_ROOT / "predictions"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)
PRED_ROOT.mkdir(parents=True, exist_ok=True)

required = [
    N06_ROOT / "06_FINALIZED.txt",
    N06_ROOT / "06_final_ml_dataset_index.csv",
    N06_ROOT / "06_normalization_train_only.json",
    N06_ROOT / "06_dataset_schema.json",
    N06_ROOT / "06_summary.json",
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise RuntimeError("Notebook 06 handoff incomplete. Missing:\n" + "\n".join(missing))

index06 = pd.read_csv(N06_ROOT / "06_final_ml_dataset_index.csv")
schema06 = json.loads((N06_ROOT / "06_dataset_schema.json").read_text())
norm06 = json.loads((N06_ROOT / "06_normalization_train_only.json").read_text())
summary06 = json.loads((N06_ROOT / "06_summary.json").read_text())

index06["sample_id"] = index06["sample_id"].astype(str)
index06["iid_split"] = index06["iid_split"].astype(str).str.lower()
index06["architecture"] = index06["architecture"].astype(str).str.lower()

if len(index06) != EXPECTED_N or not index06.sample_id.is_unique:
    raise RuntimeError("Notebook 06 index identity/count check failed.")
if index06.iid_split.value_counts().to_dict() != EXPECTED_SPLITS:
    raise RuntimeError(f"Frozen split mismatch: {index06.iid_split.value_counts().to_dict()}")

for k, col in [("node","node_feature_dim"),("edge","edge_feature_dim"),
               ("graph","graph_feature_dim"),("target","target_dim")]:
    vals = set(index06[col].astype(int).tolist())
    if vals != {EXPECTED_DIMS[k]}:
        raise RuntimeError(f"{k} dimension mismatch: {vals}")

bad_paths = [p for p in index06.graph_file.astype(str) if not Path(p).is_file()]
if bad_paths:
    raise RuntimeError(f"{len(bad_paths)} Notebook-06 graph files are missing. First: {bad_paths[0]}")

# Install/import PyG only after the handoff is proven valid.
import torch
try:
    import torch_geometric
except Exception:
    print("Installing torch-geometric...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"])
    import torch_geometric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*88)
print("NOTEBOOK 07 PREFLIGHT")
print("="*88)
print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory/1024**3:.1f} GB")
    print("✓ GPU detected — SAFE TO START TRAINING PIPELINE")
else:
    print("⚠️ CUDA GPU NOT DETECTED.")
    print("Switch Colab Runtime → Change runtime type → GPU before full training.")
    raise RuntimeError("GPU required for Notebook 07 full training.")

print("✓ Notebook 06 finalized handoff verified")
print("✓ 110 graph files present")
print("✓ frozen 70/19/21 split preserved")

# ============================================================
# MODULE 2 — DATA LOADER + TARGET INVERSE TRANSFORM
# ============================================================
from torch_geometric.data import Data

def _norm_array(key):
    # Notebook 06 JSON uses explicit target_log_mean/std fields.
    v = norm06[key]
    return np.asarray(v, dtype=np.float64)

TARGET_LOG_MEAN = _norm_array("target_log_mean")
TARGET_LOG_STD = _norm_array("target_log_std")

def inverse_target_np(y_norm):
    y_norm = np.asarray(y_norm, dtype=np.float64)
    y_log = y_norm * TARGET_LOG_STD + TARGET_LOG_MEAN
    return np.expm1(y_log)

def load_graph_row(row, include_y_phys=True):
    with np.load(str(row.graph_file), allow_pickle=False) as z:
        data = Data(
            x=torch.from_numpy(z["x"]).float(),
            edge_index=torch.from_numpy(z["edge_index"]).long(),
            edge_attr=torch.from_numpy(z["edge_attr"]).float(),
            y=torch.from_numpy(z["y"]).float(),
        )
        data.graph_attr = torch.from_numpy(z["graph_attr"]).float().reshape(1, -1)
        if include_y_phys:
            data.y_phys = torch.from_numpy(z["y_phys"]).float()
    data.sample_id = str(row.sample_id)
    data.architecture = str(row.architecture)
    data.iid_split = str(row.iid_split)
    return data

train_rows = index06[index06.iid_split == "train"].reset_index(drop=True)
val_rows = index06[index06.iid_split == "validation"].reset_index(drop=True)

# IMPORTANT: test rows are intentionally not materialized here.
if len(train_rows) != 70 or len(val_rows) != 19:
    raise RuntimeError("Train/validation split changed.")

# Read one graph and prove tensor shapes.
d0 = load_graph_row(train_rows.iloc[0])
assert d0.x.shape[1] == 8
assert d0.edge_attr.shape[1] == 9
assert d0.graph_attr.shape == (1,12)
assert d0.y.shape[1] == 2
assert d0.edge_index.shape[0] == 2
assert torch.isfinite(d0.x).all() and torch.isfinite(d0.edge_attr).all()
assert torch.isfinite(d0.y).all()

print("✓ Loader smoke test PASS")
print(d0)
print("✓ Test split remains untouched during model development")

# ============================================================
# MODULE 4 — FORWARD/BACKWARD SMOKE TEST (NO EXPENSIVE TRAINING YET)
# ============================================================
try:
    from .models import EdgeAwareGNN, MODEL_FACTORIES
except ImportError:
    from models import EdgeAwareGNN, MODEL_FACTORIES
seed_everything()
smoke = load_graph_row(train_rows.iloc[0]).to(DEVICE)

for name, factory in MODEL_FACTORIES.items():
    model = factory().to(DEVICE)
    model.train()
    pred = model(smoke)
    if pred.shape != smoke.y.shape:
        raise RuntimeError(f"{name}: output shape {pred.shape} != target {smoke.y.shape}")
    loss = F.mse_loss(pred, smoke.y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name}: non-finite smoke loss")
    loss.backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    if not grad_ok:
        raise RuntimeError(f"{name}: gradient smoke test failed")
    print(f"✓ {name:14s} forward/backward PASS | initial MSE={loss.item():.6f}")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("✓ ALL MODEL SMOKE TESTS PASS")
print("✓ SAFE TO RUN TINY-OVERFIT GATE")

# ============================================================
# MODULE 5 — TINY-OVERFIT GATE FOR PROPOSED MODEL
# ============================================================
# A healthy model/pipeline should strongly reduce training loss on two fixed graphs.
seed_everything()
tiny_data = [load_graph_row(train_rows.iloc[i]).to(DEVICE) for i in range(TINY_GRAPHS)]
tiny_model = EdgeAwareGNN().to(DEVICE)
tiny_opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3, weight_decay=0.0)

def mean_tiny_loss():
    tiny_model.eval()
    vals=[]
    with torch.no_grad():
        for d in tiny_data:
            vals.append(F.mse_loss(tiny_model(d), d.y).item())
    return float(np.mean(vals))

initial_tiny = mean_tiny_loss()
for step in range(1, TINY_MAX_STEPS + 1):
    tiny_model.train()
    tiny_opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for d in tiny_data:
        loss_sum = loss_sum + F.mse_loss(tiny_model(d), d.y)
    loss = loss_sum / len(tiny_data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 5.0)
    tiny_opt.step()
    if step % 50 == 0:
        print(f"step {step:3d} | tiny MSE={mean_tiny_loss():.6f}")

final_tiny = mean_tiny_loss()
ratio = final_tiny / max(initial_tiny, 1e-12)
print(f"Initial tiny MSE: {initial_tiny:.6f}")
print(f"Final tiny MSE:   {final_tiny:.6f}")
print(f"Fraction remaining: {ratio:.3f}")

if not np.isfinite(final_tiny) or ratio >= TINY_REQUIRED_FRACTION:
    raise RuntimeError(
        "Tiny-overfit gate FAILED. Do not spend compute on full training. "
        f"Required fraction < {TINY_REQUIRED_FRACTION}; got {ratio:.3f}."
    )

print("✓ TINY-OVERFIT GATE PASS")
print("✓ SAFE TO START FULL TRAINING")
del tiny_model, tiny_data
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# ============================================================
# MODULE 6 — TRAINING + VALIDATION FUNCTIONS
# ============================================================
def shuffled_rows(df, rng):
    idx = np.arange(len(df))
    rng.shuffle(idx)
    return df.iloc[idx]

@torch.no_grad()
def evaluate_normalized_mse(model, rows):
    model.eval()
    sse = 0.0
    n = 0
    for _, row in rows.iterrows():
        d = load_graph_row(row, include_y_phys=False).to(DEVICE)
        pred = model(d)
        diff = pred - d.y
        sse += torch.square(diff).sum().item()
        n += diff.numel()
    return sse / max(n, 1)

def train_one_model(name, factory):
    seed_everything()
    model = factory().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )
    rng = np.random.default_rng(SEED)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []
    ckpt = CKPT_ROOT / f"{name}_best.pt"
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        train_sse = 0.0
        train_n = 0
        accum = 0

        for j, (_, row) in enumerate(shuffled_rows(train_rows, rng).iterrows(), 1):
            d = load_graph_row(row, include_y_phys=False).to(DEVICE)
            pred = model(d)
            loss = F.mse_loss(pred, d.y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite training loss at epoch {epoch}")
            (loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            train_sse += torch.square(pred.detach() - d.y).sum().item()
            train_n += d.y.numel()

            if accum == GRAD_ACCUM_STEPS or j == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0

        train_mse = train_sse / train_n
        val_mse = evaluate_normalized_mse(model, val_rows)
        scheduler.step(val_mse)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch":epoch, "train_mse":train_mse,
                        "val_mse":val_mse, "lr":lr_now})

        improved = val_mse < (best_val - MIN_DELTA)
        if improved:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_name": name,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_mse": best_val,
                "seed": SEED,
                "config": {
                    "hidden_dim":HIDDEN_DIM, "num_layers":NUM_LAYERS,
                    "dropout":DROPOUT, "lr":LR, "weight_decay":WEIGHT_DECAY,
                    "grad_accum_steps":GRAD_ACCUM_STEPS,
                }
            }, ckpt)
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"{name:14s} epoch {epoch:3d} | train {train_mse:.5f} "
                  f"| val {val_mse:.5f} | best {best_val:.5f}@{best_epoch} "
                  f"| lr {lr_now:.2e}")

        if bad_epochs >= PATIENCE:
            print(f"{name}: early stopping at epoch {epoch}")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(OUT_ROOT / f"{name}_history.csv", index=False)
    elapsed = time.time() - t0
    print(f"✓ {name} complete | best val={best_val:.6f} at epoch {best_epoch} "
          f"| {elapsed/60:.1f} min")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"model":name, "best_val_mse":best_val, "best_epoch":best_epoch,
            "train_minutes":elapsed/60, "checkpoint":str(ckpt)}

print("✓ Training functions ready")

# ============================================================
# MODULE 7 — FULL TRAINING: CONTROLLED BASELINES + PROPOSED MODEL
# ============================================================
# No test data are accessed in this module.
results = []
for model_name in ["NodeMLP", "GraphSAGE", "EdgeAwareGNN"]:
    print("\n" + "="*88)
    print("TRAINING", model_name)
    print("="*88)
    results.append(train_one_model(model_name, MODEL_FACTORIES[model_name]))

val_results = pd.DataFrame(results).sort_values("best_val_mse").reset_index(drop=True)
display(val_results)

val_results.to_csv(OUT_ROOT / "07_validation_model_comparison.csv", index=False)
print("✓ Validation comparison saved")
print("✓ Test split has still not been evaluated")

# ============================================================
# MODULE 8 — FREEZE MODEL SELECTION BEFORE TOUCHING TEST SET
# ============================================================
# The proposed EdgeAwareGNN is the scientific primary model. Validation results
# are used to freeze its best epoch/checkpoint and to compare against baselines.
# We do NOT replace the proposed model with a baseline merely because of test results.
proposed_row = val_results[val_results.model == "EdgeAwareGNN"]
if len(proposed_row) != 1:
    raise RuntimeError("Expected exactly one EdgeAwareGNN validation result.")

SELECTED_MODEL = "EdgeAwareGNN"
SELECTED_CHECKPOINT = Path(proposed_row.iloc[0].checkpoint)

if not SELECTED_CHECKPOINT.is_file():
    raise RuntimeError("Selected proposed-model checkpoint is missing.")

freeze = {
    "selected_model": SELECTED_MODEL,
    "selection_basis": "pre-specified primary edge-aware GNN; best epoch selected on validation only",
    "selected_checkpoint": str(SELECTED_CHECKPOINT),
    "validation_best_mse": float(proposed_row.iloc[0].best_val_mse),
    "validation_best_epoch": int(proposed_row.iloc[0].best_epoch),
    "test_access_before_freeze": False,
    "seed": SEED,
}
(OUT_ROOT / "07_MODEL_SELECTION_FROZEN.json").write_text(json.dumps(freeze, indent=2))

print("="*88)
print("MODEL SELECTION FROZEN")
print("="*88)
print(json.dumps(freeze, indent=2))
print("✓ Only now is final test evaluation authorized.")

# ============================================================
# MODULE 9 — ONE-TIME HELD-OUT TEST EVALUATION
# ============================================================
test_rows = index06[index06.iid_split == "test"].reset_index(drop=True)
if len(test_rows) != 21:
    raise RuntimeError("Expected exactly 21 held-out test graphs.")

def load_best_model(name):
    model = MODEL_FACTORIES[name]().to(DEVICE)
    payload = torch.load(CKPT_ROOT / f"{name}_best.pt", map_location=DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

def metric_bundle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.max(y_true) - np.min(y_true))
    nrmse_range = rmse / denom if denom > 0 else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0,1])
    else:
        pearson_r = np.nan
    return {"RMSE":rmse, "MAE":mae, "NRMSE_range":nrmse_range,
            "R2":r2, "Pearson_r":pearson_r}

all_test_metrics = []
per_graph_records = []
prediction_files = []

# Evaluate all three frozen validation-selected checkpoints once for a fair test comparison.
for name in ["NodeMLP", "GraphSAGE", "EdgeAwareGNN"]:
    model = load_best_model(name)
    true_all, pred_all = [], []
    infer_times = []

    with torch.no_grad():
        for _, row in test_rows.iterrows():
            d = load_graph_row(row, include_y_phys=True).to(DEVICE)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_norm = model(d)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            pred_phys = inverse_target_np(pred_norm.detach().cpu().numpy())
            true_phys = d.y_phys.detach().cpu().numpy().astype(np.float64)
            true_all.append(true_phys)
            pred_all.append(pred_phys)
            infer_times.append(infer_ms)

            rec = {
                "model":name, "sample_id":row.sample_id,
                "architecture":row.architecture, "num_nodes":int(true_phys.shape[0]),
                "inference_ms":infer_ms
            }
            for k, target in enumerate(TARGET_NAMES):
                mb = metric_bundle(true_phys[:,k], pred_phys[:,k])
                for mk,mv in mb.items():
                    rec[f"{target}_{mk}"] = mv
            per_graph_records.append(rec)

            if name == "EdgeAwareGNN":
                p = PRED_ROOT / f"{row.sample_id}_EdgeAwareGNN_test_predictions.npz"
                np.savez_compressed(
                    p,
                    y_true_phys=true_phys.astype(np.float32),
                    y_pred_phys=pred_phys.astype(np.float32),
                    sample_id=np.asarray(str(row.sample_id)),
                    architecture=np.asarray(str(row.architecture)),
                )
                prediction_files.append(str(p))

    Y = np.concatenate(true_all, axis=0)
    P = np.concatenate(pred_all, axis=0)
    for k, target in enumerate(TARGET_NAMES):
        mb = metric_bundle(Y[:,k], P[:,k])
        all_test_metrics.append({"model":name, "target":target, **mb,
                                 "mean_inference_ms_per_graph":float(np.mean(infer_times)),
                                 "median_inference_ms_per_graph":float(np.median(infer_times))})
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

test_metrics = pd.DataFrame(all_test_metrics)
per_graph_metrics = pd.DataFrame(per_graph_records)
display(test_metrics)

test_metrics.to_csv(OUT_ROOT / "07_final_test_metrics.csv", index=False)
per_graph_metrics.to_csv(OUT_ROOT / "07_test_metrics_per_graph.csv", index=False)

print("✓ One-time held-out test evaluation complete")
print("✓ Proposed-model test predictions saved for Notebook 08")

# ============================================================
# MODULE 10 — ARCHITECTURE-WISE PROPOSED-MODEL TEST SUMMARY
# ============================================================
edge_pg = per_graph_metrics[per_graph_metrics.model == "EdgeAwareGNN"].copy()

arch_rows = []
for arch, grp in edge_pg.groupby("architecture"):
    row = {"architecture":arch, "n_graphs":len(grp)}
    for target in TARGET_NAMES:
        for metric in ["RMSE","MAE","NRMSE_range","R2","Pearson_r"]:
            col = f"{target}_{metric}"
            row[f"{target}_{metric}_mean_per_graph"] = float(grp[col].mean())
    row["inference_ms_mean"] = float(grp["inference_ms"].mean())
    arch_rows.append(row)

arch_summary = pd.DataFrame(arch_rows)
display(arch_summary)
arch_summary.to_csv(OUT_ROOT / "07_edgeaware_architecture_test_summary.csv", index=False)

print("✓ Architecture-wise summary saved")

# ============================================================
# MODULE 11 — FINALIZE NOTEBOOK-07 HANDOFF
# ============================================================
required_outputs = [
    OUT_ROOT / "07_validation_model_comparison.csv",
    OUT_ROOT / "07_MODEL_SELECTION_FROZEN.json",
    OUT_ROOT / "07_final_test_metrics.csv",
    OUT_ROOT / "07_test_metrics_per_graph.csv",
    OUT_ROOT / "07_edgeaware_architecture_test_summary.csv",
]
for p in required_outputs:
    if not p.is_file():
        raise RuntimeError(f"Missing required Notebook-07 output: {p}")

for name in MODEL_FACTORIES:
    p = CKPT_ROOT / f"{name}_best.pt"
    if not p.is_file():
        raise RuntimeError(f"Missing best checkpoint: {p}")

summary = {
    "dataset_n": 110,
    "split_counts": EXPECTED_SPLITS,
    "seed": SEED,
    "models": list(MODEL_FACTORIES.keys()),
    "primary_model": "EdgeAwareGNN",
    "node_feature_dim": 8,
    "edge_feature_dim": 9,
    "graph_feature_dim": 12,
    "target_dim": 2,
    "targets": TARGET_NAMES,
    "selection": freeze,
    "test_evaluated_once_after_freeze": True,
    "prediction_files_n": len(prediction_files),
}
(OUT_ROOT / "07_summary.json").write_text(json.dumps(summary, indent=2))
(OUT_ROOT / "07_FINALIZED.txt").write_text(
    "NOTEBOOK 07 FINALIZED\n"
    "Training, validation-only checkpoint selection, and one-time held-out test evaluation complete.\n"
    "SAFE TO PROCEED TO NOTEBOOK 08 — EVALUATION, ABLATIONS, AND FIGURES\n"
)

print("="*88)
print("NOTEBOOK 07 FINALIZATION GATE")
print("="*88)
print("✓ 3 controlled models trained on identical frozen training split")
print("✓ validation-only early stopping/checkpoint selection")
print("✓ primary EdgeAwareGNN frozen before test access")
print("✓ held-out test evaluated once after freeze")
print("✓ physical-unit metrics saved")
print("✓ proposed-model per-node test predictions saved")
print("✓ NOTEBOOK 07 FINALIZED")
print("✓ SAFE TO PROCEED TO NOTEBOOK 08 — EVALUATION, ABLATIONS, AND FIGURES")

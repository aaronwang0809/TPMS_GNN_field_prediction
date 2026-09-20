#!/usr/bin/env python3
"""Assemble, normalize, validate, and freeze the ML graph dataset.

"""
import os, json, math, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

EXPECTED_N = 110
EXPECTED_SPLITS = {"train": 70, "validation": 19, "test": 21}
EXPECTED_ARCH = {"gyroid": 39, "diamond": 38, "primitive": 33}

AUTHORIZED_TARGETS = ["von_mises_stress", "equivalent_strain"]
EXPECTED_MAPPING_K = 32
EXPECTED_IDW_POWER = 2.0
EXPECTED_MESH_LABEL = "S250k"

ARCH_ORDER = ["gyroid", "diamond", "primitive"]
KIND_ORDER = ["junction", "endpoint"]

np.random.seed(2026)


PROJECT_ROOT = Path(
    os.environ.get("TPMS_PROJECT_ROOT", Path.cwd() / "TPMS_IEEE_BIGDATA")
).expanduser().resolve()
N53_ROOT = PROJECT_ROOT / "Stage05_3"
OUT_ROOT = PROJECT_ROOT / "Stage06"
GRAPH_ROOT = OUT_ROOT / "graphs"
GRAPH_ROOT.mkdir(parents=True, exist_ok=True)

required_53 = [
    N53_ROOT / "05_3_FINALIZED.txt",
    N53_ROOT / "05_3_verified_scalar_dataset_index.csv",
    N53_ROOT / "05_3_summary.json",
]
missing = [str(p) for p in required_53 if not p.is_file()]
if missing:
    raise RuntimeError(
        "Stage 05.3 finalization handoff is incomplete. Missing:\n" +
        "\n".join(missing)
    )

summary53 = json.loads((N53_ROOT / "05_3_summary.json").read_text())

if int(summary53.get("verified_n", -1)) != EXPECTED_N:
    raise RuntimeError(f"05.3 verified_n mismatch: {summary53.get('verified_n')}")
if int(summary53.get("missing_or_in_progress_n", -1)) != 0:
    raise RuntimeError("05.3 still reports missing/in-progress samples.")
if int(summary53.get("invalid_n", -1)) != 0:
    raise RuntimeError("05.3 reports invalid checkpoints.")
if bool(summary53.get("post_qc_resplit_authorized", True)):
    raise RuntimeError("05.3 unexpectedly authorizes post-QC resplitting.")
if list(summary53.get("authorized_targets", [])) != AUTHORIZED_TARGETS:
    raise RuntimeError("05.3 target authorization mismatch.")

index53 = pd.read_csv(N53_ROOT / "05_3_verified_scalar_dataset_index.csv")
index53["sample_id"] = index53["sample_id"].astype(str)
index53["architecture"] = index53["architecture"].astype(str).str.lower()
index53["iid_split"] = index53["iid_split"].astype(str).str.lower()

if len(index53) != EXPECTED_N or not index53.sample_id.is_unique:
    raise RuntimeError("05.3 verified index is not exactly 110 unique samples.")
if index53.iid_split.value_counts().to_dict() != EXPECTED_SPLITS:
    raise RuntimeError(
        f"Frozen split mismatch: {index53.iid_split.value_counts().to_dict()}"
    )
if index53.architecture.value_counts().to_dict() != EXPECTED_ARCH:
    raise RuntimeError(
        f"Architecture count mismatch: {index53.architecture.value_counts().to_dict()}"
    )

MANIFEST_NAME = "03_FINAL_population_120_WITH_SPLITS.csv"
DESIGN_COLS = [
    "sample_id", "architecture", "iid_split",
    "target_relative_density", "cell_size_mm",
    "grading_mode", "grading_amplitude",
    "phase_x_rad", "phase_y_rad", "phase_z_rad",
]

manifest_candidates = []
for root in [PROJECT_ROOT, PROJECT_ROOT.parent]:
    try:
        manifest_candidates.extend(root.rglob(MANIFEST_NAME))
    except Exception:
        pass

manifest_candidates = list(dict.fromkeys(
    p for p in manifest_candidates if p.is_file()
))

if not manifest_candidates:
    raise RuntimeError(
        f"Authoritative design manifest not found: {MANIFEST_NAME}\n"
        "Place the original manifest under the project root or its parent directory."
    )

manifest_candidates.sort(
    key=lambda p: (0 if str(p).startswith(str(PROJECT_ROOT)) else 1, len(str(p)))
)
MANIFEST_FILE = manifest_candidates[0]
manifest = pd.read_csv(MANIFEST_FILE)

missing_design_cols = set(DESIGN_COLS) - set(manifest.columns)
if missing_design_cols:
    raise RuntimeError(
        "Authoritative manifest is missing required design columns: " +
        ", ".join(sorted(missing_design_cols))
    )

manifest = manifest[DESIGN_COLS].copy()
manifest["sample_id"] = manifest["sample_id"].astype(str)
manifest["architecture"] = manifest["architecture"].astype(str).str.lower()
manifest["iid_split"] = manifest["iid_split"].astype(str).str.lower()

if len(manifest) != 120 or not manifest.sample_id.is_unique:
    raise RuntimeError("Authoritative Stage-03.1 manifest identity check failed.")

identity = index53[["sample_id","architecture","iid_split"]].merge(
    manifest[["sample_id","architecture","iid_split"]],
    on="sample_id", how="left", suffixes=("_05_3","_03_1"), validate="one_to_one"
)
if identity[["architecture_03_1","iid_split_03_1"]].isna().any().any():
    raise RuntimeError("At least one verified 05.3 sample is absent from the 03.1 manifest.")
if not (identity["architecture_05_3"] == identity["architecture_03_1"]).all():
    raise RuntimeError("Architecture mismatch between 05.3 and Stage-03.1 manifest.")
if not (identity["iid_split_05_3"] == identity["iid_split_03_1"]).all():
    raise RuntimeError("iid_split mismatch between 05.3 and Stage-03.1 manifest.")

covariate_cols = [
    "sample_id", "cell_size_mm", "grading_mode", "grading_amplitude",
    "phase_x_rad", "phase_y_rad", "phase_z_rad",
]
rho_check = index53[["sample_id","target_relative_density"]].merge(
    manifest[["sample_id","target_relative_density"]],
    on="sample_id", how="left", suffixes=("_05_3","_03_1"), validate="one_to_one"
)
if not np.allclose(
    rho_check["target_relative_density_05_3"].to_numpy(float),
    rho_check["target_relative_density_03_1"].to_numpy(float),
    rtol=0, atol=1e-12, equal_nan=False
):
    raise RuntimeError("target_relative_density mismatch between 05.3 and 03.1 manifest.")

index53 = index53.merge(
    manifest[covariate_cols],
    on="sample_id", how="left", validate="one_to_one"
)

if index53[covariate_cols[1:]].isna().any().any():
    bad = index53.loc[
        index53[covariate_cols[1:]].isna().any(axis=1),
        ["sample_id"] + covariate_cols[1:]
    ]
    display(bad)
    raise RuntimeError("Missing design covariates after authoritative manifest merge.")


NODE_RAW_CONT = ["x", "y", "z", "degree", "region_voxels", "target_relative_density"]
NODE_BINARY = ["is_junction", "is_endpoint"]

EDGE_RAW = [
    "path_length", "chord_length", "tortuosity",
    "sin_theta", "cos_theta", "sin_phi", "cos_phi",
    "local_thickness_mean", "local_thickness_min"
]

GRAPH_FEATURE_NAMES = [
    "arch_gyroid", "arch_diamond", "arch_primitive",
    "target_relative_density", "cell_size_mm", "grading_amplitude",
    "sin_phase_x", "cos_phase_x",
    "sin_phase_y", "cos_phase_y",
    "sin_phase_z", "cos_phase_z",
]

def load_npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def finite_or_raise(name, arr, sid):
    arr = np.asarray(arr)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{sid}: non-finite values in {name}")

def design_feature_vector(row):
    arch = str(row["architecture"]).lower()
    onehot = [1.0 if arch == a else 0.0 for a in ARCH_ORDER]

    vals = [
        float(row["target_relative_density"]),
        float(row["cell_size_mm"]),
        float(row["grading_amplitude"]),
    ]
    phases = [
        float(row["phase_x_rad"]),
        float(row["phase_y_rad"]),
        float(row["phase_z_rad"]),
    ]
    trig = []
    for p in phases:
        trig.extend([math.sin(p), math.cos(p)])

    return np.asarray(onehot + vals + trig, dtype=np.float32)

def read_verified_graph(row):
    sid = str(row["sample_id"])
    d = Path(row["checkpoint_dir"])

    required = [
        d / "SUCCESS.txt",
        d / "metadata.json",
        d / "canonical_nodes.csv",
        d / "canonical_edges.csv",
        d / "graph_targets_scalar.npz",
    ]
    miss = [p.name for p in required if not p.is_file()]
    if miss:
        raise RuntimeError(f"{sid}: missing checkpoint files: {miss}")

    meta = json.loads((d / "metadata.json").read_text())
    nodes = pd.read_csv(d / "canonical_nodes.csv")
    edges = pd.read_csv(d / "canonical_edges.csv")
    targ = load_npz_dict(d / "graph_targets_scalar.npz")

    if meta.get("checkpoint_verified") is not True:
        raise RuntimeError(f"{sid}: checkpoint_verified is not true")
    if meta.get("solver_completed") is not True or int(meta.get("ccx_returncode",-1)) != 0:
        raise RuntimeError(f"{sid}: solver completion gate failed")
    if str(meta.get("sample_id")) != sid:
        raise RuntimeError(f"{sid}: metadata sample ID mismatch")
    if str(meta.get("architecture","")).lower() != str(row["architecture"]).lower():
        raise RuntimeError(f"{sid}: architecture mismatch")
    if str(meta.get("iid_split","")).lower() != str(row["iid_split"]).lower():
        raise RuntimeError(f"{sid}: iid_split mismatch")
    if str(meta.get("mesh_label")) != EXPECTED_MESH_LABEL:
        raise RuntimeError(f"{sid}: mesh label mismatch")
    if int(meta.get("mapping_k",-1)) != EXPECTED_MAPPING_K:
        raise RuntimeError(f"{sid}: mapping k mismatch")
    if not math.isclose(float(meta.get("idw_power",np.nan)),
                        EXPECTED_IDW_POWER, abs_tol=1e-12):
        raise RuntimeError(f"{sid}: IDW power mismatch")
    if list(meta.get("authorized_targets", [])) != AUTHORIZED_TARGETS:
        raise RuntimeError(f"{sid}: authorized target mismatch")
    if meta.get("full_tensor_targets_authorized") is not False:
        raise RuntimeError(f"{sid}: tensor targets unexpectedly authorized")

    required_node_cols = {"node","x","y","z","kind","region_voxels","degree"}
    required_edge_cols = {
        "source","target","path_length","chord_length","tortuosity",
        "theta","phi","local_thickness_mean","local_thickness_min","voxel_count"
    }
    if not required_node_cols.issubset(nodes.columns):
        raise RuntimeError(f"{sid}: canonical node columns mismatch")
    if not required_edge_cols.issubset(edges.columns):
        raise RuntimeError(f"{sid}: canonical edge columns mismatch")
    if set(nodes["kind"].astype(str).unique()) - set(KIND_ORDER):
        raise RuntimeError(f"{sid}: unexpected node kind(s)")

    n = len(nodes)
    if int(meta.get("graph_nodes",-1)) != n:
        raise RuntimeError(f"{sid}: graph node count mismatch")
    if int(meta.get("graph_edges",-1)) != len(edges):
        raise RuntimeError(f"{sid}: graph edge count mismatch")

    node_ids = nodes["node"].to_numpy(dtype=np.int64)
    if not np.array_equal(node_ids, np.arange(n, dtype=np.int64)):
        raise RuntimeError(f"{sid}: node IDs are not contiguous 0..N-1")

    src = edges["source"].to_numpy(dtype=np.int64)
    dst = edges["target"].to_numpy(dtype=np.int64)
    if len(src) == 0 or src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
        raise RuntimeError(f"{sid}: edge endpoint index out of range")
    if np.any(src == dst):
        raise RuntimeError(f"{sid}: self-loop found in canonical raw graph")

    y_vm = np.asarray(targ["y_vm"], dtype=np.float32)
    y_eq = np.asarray(targ["y_eq_strain"], dtype=np.float32)
    if y_vm.shape != (n,) or y_eq.shape != (n,):
        raise RuntimeError(f"{sid}: target length does not match graph nodes")
    if int(np.asarray(targ["mapping_k"]).item()) != EXPECTED_MAPPING_K:
        raise RuntimeError(f"{sid}: target payload mapping_k mismatch")
    if not math.isclose(float(np.asarray(targ["idw_power"]).item()),
                        EXPECTED_IDW_POWER, abs_tol=1e-12):
        raise RuntimeError(f"{sid}: target payload idw_power mismatch")
    finite_or_raise("y_vm", y_vm, sid)
    finite_or_raise("y_eq_strain", y_eq, sid)
    if np.any(y_vm < 0) or np.any(y_eq < 0):
        raise RuntimeError(f"{sid}: negative scalar target found")

    node_raw_cont = np.column_stack([
        nodes["x"].to_numpy(np.float32),
        nodes["y"].to_numpy(np.float32),
        nodes["z"].to_numpy(np.float32),
        nodes["degree"].to_numpy(np.float32),
        nodes["region_voxels"].to_numpy(np.float32),
        np.full(n, float(row["target_relative_density"]), dtype=np.float32),
    ]).astype(np.float32)

    node_binary = np.column_stack([
        (nodes["kind"].astype(str).to_numpy() == "junction").astype(np.float32),
        (nodes["kind"].astype(str).to_numpy() == "endpoint").astype(np.float32),
    ]).astype(np.float32)

    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ]).astype(np.int64)

    theta = edges["theta"].to_numpy(np.float32)
    phi = edges["phi"].to_numpy(np.float32)
    edge_raw_undirected = np.column_stack([
        edges["path_length"].to_numpy(np.float32),
        edges["chord_length"].to_numpy(np.float32),
        edges["tortuosity"].to_numpy(np.float32),
        np.sin(theta), np.cos(theta),
        np.sin(phi), np.cos(phi),
        edges["local_thickness_mean"].to_numpy(np.float32),
        edges["local_thickness_min"].to_numpy(np.float32),
    ]).astype(np.float32)
    edge_raw = np.concatenate([edge_raw_undirected, edge_raw_undirected], axis=0)

    finite_or_raise("node_raw_cont", node_raw_cont, sid)
    finite_or_raise("node_binary", node_binary, sid)
    finite_or_raise("edge_raw", edge_raw, sid)

    graph_raw = design_feature_vector(row)
    finite_or_raise("graph_raw", graph_raw, sid)

    return {
        "sample_id": sid,
        "architecture": str(row["architecture"]).lower(),
        "iid_split": str(row["iid_split"]).lower(),
        "node_raw_cont": node_raw_cont,
        "node_binary": node_binary,
        "edge_index": edge_index,
        "edge_raw": edge_raw,
        "graph_raw": graph_raw,
        "y_vm": y_vm,
        "y_eq_strain": y_eq,
        "num_nodes": n,
        "num_edges_undirected": len(edges),
        "num_edges_directed": edge_index.shape[1],
    }


t0 = time.time()
raw_graphs = []
errors = []

for i, (_, row) in enumerate(index53.iterrows(), 1):
    try:
        g = read_verified_graph(row)
        raw_graphs.append(g)
        if i == 1 or i % 10 == 0 or i == EXPECTED_N:
            print(
                f"[{i:3d}/{EXPECTED_N}] {g['sample_id']} "
                f"{g['architecture']:9s} {g['iid_split']:10s} "
                f"N={g['num_nodes']:,} E={g['num_edges_undirected']:,}"
            )
    except Exception as e:
        errors.append((str(row["sample_id"]), repr(e)))

if errors:
    display(pd.DataFrame(errors, columns=["sample_id","error"]))
    raise RuntimeError(f"{len(errors)} graph(s) failed Stage-06 validation.")

if len(raw_graphs) != EXPECTED_N:
    raise RuntimeError(f"Expected 110 validated graphs; got {len(raw_graphs)}")

ids = [g["sample_id"] for g in raw_graphs]
if len(ids) != len(set(ids)):
    raise RuntimeError("Duplicate sample IDs detected in assembled dataset.")


train_graphs = [g for g in raw_graphs if g["iid_split"] == "train"]
val_graphs = [g for g in raw_graphs if g["iid_split"] == "validation"]
test_graphs = [g for g in raw_graphs if g["iid_split"] == "test"]

if (len(train_graphs), len(val_graphs), len(test_graphs)) != (70,19,21):
    raise RuntimeError("Frozen split counts changed during assembly.")

def streaming_mean_std(arrays, dim):
    count = 0
    s = np.zeros(dim, dtype=np.float64)
    ss = np.zeros(dim, dtype=np.float64)
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        count += a.shape[0]
        s += a.sum(axis=0)
        ss += np.square(a).sum(axis=0)
    if count == 0:
        raise RuntimeError("Cannot fit normalization on empty collection.")
    mean = s / count
    var = np.maximum(ss / count - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), int(count)

node_mean, node_std, n_train_nodes = streaming_mean_std(
    (g["node_raw_cont"] for g in train_graphs), len(NODE_RAW_CONT)
)
edge_mean, edge_std, n_train_edges = streaming_mean_std(
    (g["edge_raw"] for g in train_graphs), len(EDGE_RAW)
)

graph_train = np.stack([g["graph_raw"] for g in train_graphs])
graph_mean = graph_train.mean(axis=0).astype(np.float32)
graph_std = graph_train.std(axis=0).astype(np.float32)
graph_mean[:3] = 0.0
graph_std[:3] = 1.0
graph_std[graph_std < 1e-12] = 1.0

train_vm = np.concatenate([g["y_vm"] for g in train_graphs]).astype(np.float64)
train_eq = np.concatenate([g["y_eq_strain"] for g in train_graphs]).astype(np.float64)

if np.any(train_vm < 0) or np.any(train_eq < 0):
    raise RuntimeError("Targets must be nonnegative before log1p transform.")

vm_log = np.log1p(train_vm)
eq_log = np.log1p(train_eq)

target_log_mean = np.asarray([vm_log.mean(), eq_log.mean()], dtype=np.float32)
target_log_std = np.asarray([vm_log.std(), eq_log.std()], dtype=np.float32)
target_log_std[target_log_std < 1e-12] = 1.0

normalization = {
    "node_cont_names": NODE_RAW_CONT,
    "node_mean": node_mean.tolist(),
    "node_std": node_std.tolist(),
    "edge_names": EDGE_RAW,
    "edge_mean": edge_mean.tolist(),
    "edge_std": edge_std.tolist(),
    "graph_names": GRAPH_FEATURE_NAMES,
    "graph_mean": graph_mean.tolist(),
    "graph_std": graph_std.tolist(),
    "target_names": AUTHORIZED_TARGETS,
    "target_transform": "log1p_then_zscore",
    "target_log_mean": target_log_mean.tolist(),
    "target_log_std": target_log_std.tolist(),
    "fit_split": "train_only",
    "fit_graphs": 70,
    "fit_nodes": n_train_nodes,
    "fit_directed_edges": n_train_edges,
}


def zscore(a, mean, std):
    return ((a - mean) / std).astype(np.float32)

def transform_graph(g):
    node_cont = zscore(g["node_raw_cont"], node_mean, node_std)
    x = np.concatenate([node_cont, g["node_binary"]], axis=1).astype(np.float32)

    edge_attr = zscore(g["edge_raw"], edge_mean, edge_std)
    graph_attr = zscore(g["graph_raw"], graph_mean, graph_std)

    y_phys = np.column_stack([g["y_vm"], g["y_eq_strain"]]).astype(np.float32)
    y_log = np.log1p(y_phys.astype(np.float64))
    y = ((y_log - target_log_mean) / target_log_std).astype(np.float32)

    for name, arr in [
        ("x",x), ("edge_attr",edge_attr), ("graph_attr",graph_attr),
        ("y",y), ("y_phys",y_phys)
    ]:
        finite_or_raise(name, arr, g["sample_id"])

    return x, edge_attr, graph_attr, y, y_phys

records = []
for i, g in enumerate(raw_graphs, 1):
    x, edge_attr, graph_attr, y, y_phys = transform_graph(g)

    out = GRAPH_ROOT / f"{g['sample_id']}.npz"
    np.savez_compressed(
        out,
        x=x,
        edge_index=g["edge_index"].astype(np.int64),
        edge_attr=edge_attr,
        graph_attr=graph_attr,
        y=y,
        y_phys=y_phys,
        sample_id=np.asarray(g["sample_id"]),
        architecture=np.asarray(g["architecture"]),
        iid_split=np.asarray(g["iid_split"]),
    )

    with np.load(out, allow_pickle=False) as z:
        if z["x"].shape != x.shape:
            raise RuntimeError(f"{g['sample_id']}: saved x shape mismatch")
        if z["edge_index"].shape != g["edge_index"].shape:
            raise RuntimeError(f"{g['sample_id']}: saved edge_index shape mismatch")
        if z["edge_attr"].shape[0] != z["edge_index"].shape[1]:
            raise RuntimeError(f"{g['sample_id']}: edge feature/index mismatch")
        if z["y"].shape != (g["num_nodes"], 2):
            raise RuntimeError(f"{g['sample_id']}: saved target shape mismatch")
        if not np.isfinite(z["x"]).all() or not np.isfinite(z["edge_attr"]).all():
            raise RuntimeError(f"{g['sample_id']}: nonfinite saved features")
        if not np.isfinite(z["y"]).all() or not np.isfinite(z["y_phys"]).all():
            raise RuntimeError(f"{g['sample_id']}: nonfinite saved targets")

    records.append({
        "sample_id": g["sample_id"],
        "architecture": g["architecture"],
        "iid_split": g["iid_split"],
        "num_nodes": g["num_nodes"],
        "num_edges_undirected": g["num_edges_undirected"],
        "num_edges_directed": g["num_edges_directed"],
        "node_feature_dim": x.shape[1],
        "edge_feature_dim": edge_attr.shape[1],
        "graph_feature_dim": graph_attr.shape[0],
        "target_dim": y.shape[1],
        "graph_file": str(out),
        "target_relative_density": float(
            index53.loc[index53.sample_id == g["sample_id"], "target_relative_density"].iloc[0]
        ),
    })

dataset_index = pd.DataFrame(records)

if len(dataset_index) != EXPECTED_N:
    raise RuntimeError("Saved graph index is not exactly 110 rows.")


if set(dataset_index.sample_id) != set(index53.sample_id):
    raise RuntimeError("Stage-06 sample identity differs from Stage-05.3.")

if dataset_index.iid_split.value_counts().to_dict() != EXPECTED_SPLITS:
    raise RuntimeError("Stage-06 split counts differ from frozen split.")

if dataset_index.architecture.value_counts().to_dict() != EXPECTED_ARCH:
    raise RuntimeError("Stage-06 architecture counts differ from 05.3.")

split_sets = {
    s: set(dataset_index.loc[dataset_index.iid_split == s, "sample_id"])
    for s in ["train","validation","test"]
}
if split_sets["train"] & split_sets["validation"]:
    raise RuntimeError("Train/validation identity leakage.")
if split_sets["train"] & split_sets["test"]:
    raise RuntimeError("Train/test identity leakage.")
if split_sets["validation"] & split_sets["test"]:
    raise RuntimeError("Validation/test identity leakage.")



try:
    import torch
    try:
        from torch_geometric.data import Data
    except Exception:
        print("Installing torch-geometric (lightweight Python package)...")
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'torch-geometric'])
        from torch_geometric.data import Data

    sample_row = dataset_index.iloc[0]
    with np.load(sample_row.graph_file, allow_pickle=False) as z:
        data = Data(
            x=torch.from_numpy(z["x"]).float(),
            edge_index=torch.from_numpy(z["edge_index"]).long(),
            edge_attr=torch.from_numpy(z["edge_attr"]).float(),
            y=torch.from_numpy(z["y"]).float(),
        )
        data.y_phys = torch.from_numpy(z["y_phys"]).float()
        data.graph_attr = torch.from_numpy(z["graph_attr"]).float().unsqueeze(0)

    if data.x.shape[0] != data.y.shape[0]:
        raise RuntimeError("PyG smoke test: node/target mismatch")
    if data.edge_index.shape[1] != data.edge_attr.shape[0]:
        raise RuntimeError("PyG smoke test: edge/index mismatch")


except Exception as e:
    raise RuntimeError(f"PyG compatibility smoke test failed: {type(e).__name__}: {e}") from e

index_path = OUT_ROOT / "06_final_ml_dataset_index.csv"
norm_path = OUT_ROOT / "06_normalization_train_only.json"
schema_path = OUT_ROOT / "06_dataset_schema.json"
summary_path = OUT_ROOT / "06_summary.json"
marker_path = OUT_ROOT / "06_FINALIZED.txt"

dataset_index.to_csv(index_path, index=False)
norm_path.write_text(json.dumps(normalization, indent=2))

schema = {
    "format": "one compressed NPZ per graph",
    "node_feature_names": NODE_RAW_CONT + NODE_BINARY,
    "node_feature_dim": len(NODE_RAW_CONT) + len(NODE_BINARY),
    "edge_feature_names": EDGE_RAW,
    "edge_feature_dim": len(EDGE_RAW),
    "graph_feature_names": GRAPH_FEATURE_NAMES,
    "graph_feature_dim": len(GRAPH_FEATURE_NAMES),
    "target_names": AUTHORIZED_TARGETS,
    "target_dim": 2,
    "training_target_field": "y",
    "physical_target_field": "y_phys",
    "edge_index_convention": "directed COO; each canonical undirected edge stored in both directions",
    "normalization": "training-only statistics",
    "target_transform": "log1p_then_zscore",
    "full_tensor_targets_authorized": False,
    "post_qc_resplit": False,
}
schema_path.write_text(json.dumps(schema, indent=2))

summary06 = {
    "stage": "06",
    "status": "FINALIZED",
    "graphs": int(len(dataset_index)),
    "train": int((dataset_index.iid_split=="train").sum()),
    "validation": int((dataset_index.iid_split=="validation").sum()),
    "test": int((dataset_index.iid_split=="test").sum()),
    "by_architecture": {
        k:int(v) for k,v in dataset_index.architecture.value_counts().to_dict().items()
    },
    "total_nodes": int(dataset_index.num_nodes.sum()),
    "total_undirected_edges": int(dataset_index.num_edges_undirected.sum()),
    "total_directed_edges": int(dataset_index.num_edges_directed.sum()),
    "node_feature_dim": int(schema["node_feature_dim"]),
    "edge_feature_dim": int(schema["edge_feature_dim"]),
    "graph_feature_dim": int(schema["graph_feature_dim"]),
    "target_dim": 2,
    "normalization_fit_split": "train_only",
    "post_qc_resplit": False,
}
summary_path.write_text(json.dumps(summary06, indent=2))

saved_npz = sorted(GRAPH_ROOT.glob("*.npz"))
if len(saved_npz) != EXPECTED_N:
    raise RuntimeError(f"Expected 110 saved graph NPZs; found {len(saved_npz)}")

marker_path.write_text(
    "Stage 06 FINALIZED\n"
    "graphs=110\n"
    "train=70\nvalidation=19\ntest=21\n"
    "targets=von_mises_stress,equivalent_strain\n"
    "normalization=train_only\n"
    "post_qc_resplit=false\n"
)

print(
    f"Dataset assembly complete: {len(dataset_index)} graphs "
    f"({len(train_graphs)} train, {len(val_graphs)} validation, {len(test_graphs)} test)"
)

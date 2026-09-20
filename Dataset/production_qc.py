#!/usr/bin/env python3
"""Consolidate production checkpoints and perform dataset handoff QC.

"""
import os, json, zipfile, math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

MODE = os.environ.get("TPMS_QC_MODE", "PROGRESS").upper()
assert MODE in {"PROGRESS", "FINALIZE"}

MANIFEST_NAME = "03_FINAL_population_120_WITH_SPLITS.csv"
REQUIRED_ZIP_HINT = "03_1_FINAL_TPMS_DATASET_DESIGN.zip"

EXPECTED_TARGETS = ["von_mises_stress", "equivalent_strain"]
EXPECTED_MAPPING_K = 32
EXPECTED_IDW_POWER = 2.0
EXPECTED_MESH_LABEL = "S250k"
MAX_K32_RADIUS_P95_MM = 0.50


PROJECT_ROOT = Path(
    os.environ.get("TPMS_PROJECT_ROOT", Path.cwd() / "TPMS_IEEE_BIGDATA")
).expanduser().resolve()
MYDRIVE = PROJECT_ROOT.parent

if not PROJECT_ROOT.exists():
    raise RuntimeError(f"Project root not found: {PROJECT_ROOT}")

OUT_ROOT = PROJECT_ROOT / "Stage05_3"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def find_manifest():
    candidates = []
    for root in [PROJECT_ROOT, MYDRIVE, Path.cwd()]:
        try:
            candidates.extend(root.rglob(MANIFEST_NAME))
        except Exception:
            pass

    seen = set()
    unique = []
    for p in candidates:
        s = str(p)
        if s not in seen and p.is_file():
            seen.add(s)
            unique.append(p)
    return unique

manifest_candidates = find_manifest()

if manifest_candidates:
    MANIFEST_FILE = manifest_candidates[0]
else:
    zip_candidates = list(PROJECT_ROOT.rglob(REQUIRED_ZIP_HINT)) + list(Path.cwd().glob(REQUIRED_ZIP_HINT))
    if len(zip_candidates) != 1:
        raise RuntimeError(
            f"Could not find {MANIFEST_NAME}; place one {REQUIRED_ZIP_HINT} "
            f"under {PROJECT_ROOT} or in the current directory."
        )
    zp = zip_candidates[0]
    stage = OUT_ROOT / "manifest_stage"
    stage.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zp) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at: {bad}")
        names = zf.namelist()
        if not any(Path(n).name == MANIFEST_NAME for n in names):
            raise RuntimeError(f"Wrong ZIP: {MANIFEST_NAME} not found.")
        zf.extractall(stage)

    found = list(stage.rglob(MANIFEST_NAME))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {MANIFEST_NAME}; found {len(found)}.")
    MANIFEST_FILE = found[0]


manifest = pd.read_csv(MANIFEST_FILE)

required_cols = {
    "sample_id", "architecture", "target_relative_density",
    "cell_size_mm", "grading_mode", "grading_amplitude",
    "phase_x_rad", "phase_y_rad", "phase_z_rad", "iid_split"
}
missing_cols = required_cols - set(manifest.columns)
if missing_cols:
    raise RuntimeError("Manifest missing columns: " + ", ".join(sorted(missing_cols)))

manifest["sample_id"] = manifest["sample_id"].astype(str)
manifest["architecture"] = manifest["architecture"].astype(str).str.lower()
manifest["iid_split"] = manifest["iid_split"].astype(str).str.lower()

if len(manifest) != 120:
    raise RuntimeError(f"Expected 120 manifest rows; got {len(manifest)}.")
if not manifest["sample_id"].is_unique:
    raise RuntimeError("Manifest sample_id values are not unique.")

arch_counts = manifest.groupby("architecture").size().to_dict()
expected_archs = {"gyroid", "diamond", "primitive"}
if set(arch_counts) != expected_archs:
    raise RuntimeError(f"Architecture identity mismatch: {arch_counts}")
if any(arch_counts[a] != 40 for a in expected_archs):
    raise RuntimeError(f"Expected exactly 40 per architecture: {arch_counts}")

primitive = manifest[manifest.architecture == "primitive"]["sample_id"].tolist()
diamond   = manifest[manifest.architecture == "diamond"]["sample_id"].tolist()
gyroid    = manifest[manifest.architecture == "gyroid"]["sample_id"].tolist()

ASSIGNMENTS = {
    "05_2A": primitive[:20],
    "05_2B": primitive[20:40],
    "05_2C": diamond[:20],
    "05_2D": diamond[20:40],
    "05_2E": gyroid[:20],
    "05_2F": gyroid[20:40],
}

all_assigned = [sid for ids in ASSIGNMENTS.values() for sid in ids]
if len(all_assigned) != 120 or len(set(all_assigned)) != 120:
    raise RuntimeError("Frozen six-worker assignment is not an exact unique 120-design cover.")
if set(all_assigned) != set(manifest.sample_id):
    raise RuntimeError("Frozen worker assignments do not exactly match manifest IDs.")

assignment_rows = []
for worker, ids in ASSIGNMENTS.items():
    for pos, sid in enumerate(ids, 1):
        assignment_rows.append({
            "worker": worker,
            "worker_position": pos,
            "sample_id": sid
        })
assignment_df = pd.DataFrame(assignment_rows)

manifest_audit = manifest.merge(assignment_df, on="sample_id", how="left", validate="one_to_one")


CHECKPOINT_ROOTS = {
    worker: PROJECT_ROOT / f"Stage{worker}" / "production_checkpoints"
    for worker in ASSIGNMENTS
}

REQUIRED_SUCCESS_FILES = [
    "SUCCESS.txt",
    "metadata.json",
    "canonical_nodes.csv",
    "canonical_edges.csv",
    "mapping_k32.npz",
    "target_scalar_elements.npz",
    "graph_targets_scalar.npz",
]

def safe_json(path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"__json_error__": f"{type(e).__name__}: {e}"}

def validate_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            shapes = {k: tuple(z[k].shape) for k in keys}
            finite = {}
            for k in keys:
                arr = z[k]
                if np.issubdtype(arr.dtype, np.number):
                    finite[k] = bool(np.isfinite(arr).all())
        return True, keys, shapes, finite, ""
    except Exception as e:
        return False, [], {}, {}, f"{type(e).__name__}: {e}"

def safe_int(value, default=-999):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default

def validate_success_dir(sample_dir, expected_sid, expected_arch, expected_split):
    issues = []

    missing = [f for f in REQUIRED_SUCCESS_FILES if not (sample_dir / f).is_file()]
    if missing:
        issues.append("missing_success_files=" + ",".join(missing))

    meta = {}
    mp = sample_dir / "metadata.json"
    if mp.is_file():
        meta = safe_json(mp)
        if "__json_error__" in meta:
            issues.append("metadata_json_error=" + meta["__json_error__"])
        else:
            if str(meta.get("sample_id")) != expected_sid:
                issues.append(f"metadata_sample_id={meta.get('sample_id')}")
            if str(meta.get("architecture","")).lower() != expected_arch:
                issues.append(f"metadata_architecture={meta.get('architecture')}")
            if str(meta.get("iid_split","")).lower() != expected_split:
                issues.append(f"metadata_iid_split={meta.get('iid_split')}")
            if meta.get("checkpoint_verified") is not True:
                issues.append("checkpoint_verified_not_true")
            if meta.get("solver_completed") is not True:
                issues.append("solver_completed_not_true")
            if safe_int(meta.get("ccx_returncode")) != 0:
                issues.append(f"ccx_returncode={meta.get('ccx_returncode')}")
            if str(meta.get("mesh_label")) != EXPECTED_MESH_LABEL:
                issues.append(f"mesh_label={meta.get('mesh_label')}")
            if safe_int(meta.get("mapping_k"), -1) != EXPECTED_MAPPING_K:
                issues.append(f"mapping_k={meta.get('mapping_k')}")
            if not math.isclose(safe_float(meta.get("idw_power")),
                                EXPECTED_IDW_POWER, rel_tol=0, abs_tol=1e-12):
                issues.append(f"idw_power={meta.get('idw_power')}")
            if meta.get("full_tensor_targets_authorized") is not False:
                issues.append("full_tensor_targets_authorized_not_false")
            if list(meta.get("authorized_targets", [])) != EXPECTED_TARGETS:
                issues.append(f"authorized_targets={meta.get('authorized_targets')}")
            if safe_int(meta.get("graph_components"), -1) != 1:
                issues.append(f"graph_components={meta.get('graph_components')}")
            if safe_int(meta.get("zero_volume_tets"), -1) != 0:
                issues.append(f"zero_volume_tets={meta.get('zero_volume_tets')}")
            radius = safe_float(meta.get("k32_radius_p95_mm"))
            if not np.isfinite(radius) or radius > MAX_K32_RADIUS_P95_MM:
                issues.append(f"k32_radius_p95_mm={radius}")

    for fname in ["mapping_k32.npz", "target_scalar_elements.npz", "graph_targets_scalar.npz"]:
        p = sample_dir / fname
        if p.is_file():
            ok, keys, shapes, finite, err = validate_npz(p)
            if not ok:
                issues.append(f"{fname}_unreadable={err}")
            elif any(v is False for v in finite.values()):
                bad = [k for k,v in finite.items() if not v]
                issues.append(f"{fname}_nonfinite_keys={bad}")

    return meta, issues


manifest_lookup = manifest.set_index("sample_id").to_dict("index")
rows = []

observed_locations = {}
for worker, root in CHECKPOINT_ROOTS.items():
    if not root.exists():
        continue
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("TPMS_"):
            observed_locations.setdefault(p.name, []).append(worker)

for worker, ids in ASSIGNMENTS.items():
    root = CHECKPOINT_ROOTS[worker]
    for pos, sid in enumerate(ids, 1):
        mrow = manifest_lookup[sid]
        arch = str(mrow["architecture"]).lower()
        split = str(mrow["iid_split"]).lower()
        d = root / sid

        status = "MISSING_OR_IN_PROGRESS"
        failure_stage = ""
        failure_error = ""
        validation_issues = []
        meta = {}

        if d.exists():
            success_marker = (d / "SUCCESS.txt").is_file()
            failure_marker = (d / "FAILURE.json").is_file()

            if success_marker and failure_marker:
                status = "INVALID_CONFLICT"
                validation_issues.append("both_SUCCESS_and_FAILURE_markers")
            elif success_marker:
                meta, validation_issues = validate_success_dir(d, sid, arch, split)
                status = "VERIFIED" if not validation_issues else "INVALID_SUCCESS_CHECKPOINT"
            elif failure_marker:
                fail = safe_json(d / "FAILURE.json")
                if "__json_error__" in fail:
                    status = "INVALID_FAILURE_RECORD"
                    validation_issues.append(fail["__json_error__"])
                else:
                    status = "TERMINAL_QC_FAILURE"
                    failure_stage = str(fail.get("stage", ""))
                    failure_error = str(fail.get("error", ""))
            else:
                status = "MISSING_OR_IN_PROGRESS"

        locs = observed_locations.get(sid, [])
        wrong_locations = [w for w in locs if w != worker]
        if wrong_locations:
            validation_issues.append("also_present_in=" + ",".join(wrong_locations))
            if status == "VERIFIED":
                status = "INVALID_CROSS_WORKER_DUPLICATE"

        rows.append({
            "sample_id": sid,
            "architecture": arch,
            "iid_split": split,
            "worker": worker,
            "worker_position": pos,
            "status": status,
            "failure_stage": failure_stage,
            "failure_error": failure_error,
            "validation_issues": " | ".join(validation_issues),
            "checkpoint_dir": str(d),
            "target_relative_density": mrow.get("target_relative_density", np.nan),
            "rho_128": meta.get("rho_128", np.nan),
            "rho_abs_error_128": meta.get("rho_abs_error_128", np.nan),
            "graph_nodes": meta.get("graph_nodes", np.nan),
            "graph_edges": meta.get("graph_edges", np.nan),
            "mesh_nodes": meta.get("mesh_nodes", np.nan),
            "mesh_tets": meta.get("mesh_tets", np.nan),
            "mesh_q01": meta.get("mesh_q01", np.nan),
            "mesh_q_median": meta.get("mesh_q_median", np.nan),
            "mesh_edge_ratio_p99": meta.get("mesh_edge_ratio_p99", np.nan),
            "k32_radius_p95_mm": meta.get("k32_radius_p95_mm", np.nan),
            "target_unique_elements": meta.get("target_unique_elements", np.nan),
            "solve_minutes": meta.get("solve_minutes", np.nan),
            "runtime_minutes_total": meta.get("runtime_minutes_total", np.nan),
            "vm_graph_median_MPa": meta.get("vm_graph_median_MPa", np.nan),
            "vm_graph_p95_MPa": meta.get("vm_graph_p95_MPa", np.nan),
            "eq_graph_median": meta.get("eq_graph_median", np.nan),
            "eq_graph_p95": meta.get("eq_graph_p95", np.nan),
        })

audit = pd.DataFrame(rows)

if len(audit) != 120 or not audit.sample_id.is_unique:
    raise RuntimeError("05.3 audit did not produce exactly 120 unique manifest designs.")

print("Production status")
display(audit["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count"))

print("\nBy architecture:")
display(pd.crosstab(audit["architecture"], audit["status"]))

print("\nBy frozen iid_split:")
display(pd.crosstab(audit["iid_split"], audit["status"]))

print("\nBy architecture × frozen split (verified only):")
verified = audit[audit.status == "VERIFIED"].copy()
display(verified.groupby(["architecture","iid_split"]).size().unstack(fill_value=0))

failures = audit[audit.status == "TERMINAL_QC_FAILURE"].copy()
unfinished = audit[audit.status == "MISSING_OR_IN_PROGRESS"].copy()
invalid = audit[audit.status.str.startswith("INVALID", na=False)].copy()

def classify_failure(stage, error):
    s = (str(stage) + " " + str(error)).lower()
    if "mapping" in s or "k32" in s or "radius" in s:
        return "mapping_qc"
    if "geometry" in s or "component" in s or "rho_err" in s:
        return "geometry_qc"
    if "mesh" in s or "tetgen" in s:
        return "mesh_qc_or_meshing"
    if "ccx" in s or "calculix" in s or "solver" in s:
        return "solver"
    if "parse" in s or "stress" in s or "strain" in s:
        return "parser_or_field_output"
    return "other"

if len(failures):
    failures["failure_category"] = [
        classify_failure(s,e) for s,e in zip(failures.failure_stage, failures.failure_error)
    ]
    print("Terminal failure categories:")
    display(failures.groupby(["architecture","failure_category"]).size()
            .rename("count").reset_index())
    display(failures[[
        "sample_id","architecture","iid_split","worker",
        "failure_category","failure_stage","failure_error"
    ]])
else:
    print("No terminal QC failures currently visible.")

if len(unfinished):
    print("\nMissing/in-progress designs:")
    display(unfinished[["sample_id","architecture","iid_split","worker","worker_position"]])

if len(invalid):
    print("\nINVALID CHECKPOINTS — must be resolved before Stage 06:")
    display(invalid[["sample_id","worker","status","validation_issues"]])

summary_cols = [
    "rho_abs_error_128","graph_nodes","graph_edges","mesh_nodes","mesh_tets",
    "mesh_q01","mesh_q_median","mesh_edge_ratio_p99","k32_radius_p95_mm",
    "target_unique_elements","solve_minutes","runtime_minutes_total",
    "vm_graph_median_MPa","vm_graph_p95_MPa",
    "eq_graph_median","eq_graph_p95"
]

if len(verified):
    production_stats = verified.groupby("architecture")[summary_cols].agg(
        ["count","mean","std","min","median","max"]
    )
    print("\nVerified production statistics:")
    display(production_stats)

split_before = (
    audit.groupby(["architecture","iid_split"])
         .size().rename("original_n").reset_index()
)
split_after = (
    verified.groupby(["architecture","iid_split"])
            .size().rename("verified_n").reset_index()
)

split_attrition = split_before.merge(
    split_after, on=["architecture","iid_split"], how="left"
)
split_attrition["verified_n"] = split_attrition["verified_n"].fillna(0).astype(int)
split_attrition["excluded_or_unfinished_n"] = (
    split_attrition["original_n"] - split_attrition["verified_n"]
)
split_attrition["retention_pct"] = (
    100.0 * split_attrition["verified_n"] / split_attrition["original_n"]
)

print("FROZEN SPLIT ATTRITION AUDIT")
display(split_attrition)

overall_split = pd.DataFrame({
    "original_n": audit.groupby("iid_split").size(),
    "verified_n": verified.groupby("iid_split").size()
}).fillna(0).astype(int)
overall_split["excluded_or_unfinished_n"] = (
    overall_split["original_n"] - overall_split["verified_n"]
)
overall_split["retention_pct"] = (
    100.0 * overall_split["verified_n"] / overall_split["original_n"]
)
display(overall_split.reset_index())


audit_path = OUT_ROOT / "05_3_production_audit_120.csv"
verified_path = OUT_ROOT / "05_3_verified_scalar_dataset_index.csv"
failure_path = OUT_ROOT / "05_3_terminal_failures.csv"
attrition_path = OUT_ROOT / "05_3_split_attrition.csv"
summary_path = OUT_ROOT / "05_3_summary.json"

audit.to_csv(audit_path, index=False)
verified.to_csv(verified_path, index=False)
failures.to_csv(failure_path, index=False)
split_attrition.to_csv(attrition_path, index=False)

status_counts = audit.status.value_counts().to_dict()

summary = {
    "stage": "05.3",
    "mode": MODE,
    "manifest_file": str(MANIFEST_FILE),
    "manifest_n": int(len(manifest)),
    "assigned_n": int(len(audit)),
    "verified_n": int((audit.status == "VERIFIED").sum()),
    "terminal_qc_failure_n": int((audit.status == "TERMINAL_QC_FAILURE").sum()),
    "missing_or_in_progress_n": int((audit.status == "MISSING_OR_IN_PROGRESS").sum()),
    "invalid_n": int(audit.status.str.startswith("INVALID", na=False).sum()),
    "status_counts": {str(k): int(v) for k,v in status_counts.items()},
    "verified_by_architecture": {
        str(k): int(v) for k,v in verified.groupby("architecture").size().to_dict().items()
    },
    "verified_by_iid_split": {
        str(k): int(v) for k,v in verified.groupby("iid_split").size().to_dict().items()
    },
    "authorized_targets": EXPECTED_TARGETS,
    "full_tensor_targets_authorized": False,
    "mapping_k": EXPECTED_MAPPING_K,
    "idw_power": EXPECTED_IDW_POWER,
    "mesh_label": EXPECTED_MESH_LABEL,
    "post_qc_resplit_authorized": False,
}
summary_path.write_text(json.dumps(summary, indent=2))

print("Saved:")
for p in [audit_path, verified_path, failure_path, attrition_path, summary_path]:
    print(" ✓", p)

n_verified = int((audit.status == "VERIFIED").sum())
n_failed = int((audit.status == "TERMINAL_QC_FAILURE").sum())
n_unfinished = int((audit.status == "MISSING_OR_IN_PROGRESS").sum())
n_invalid = int(audit.status.str.startswith("INVALID", na=False).sum())

print(f"Verified successful checkpoints : {n_verified}")
print(f"Terminal predefined QC failures : {n_failed}")
print(f"Missing / still in progress      : {n_unfinished}")
print(f"Invalid/conflicting checkpoints  : {n_invalid}")
print(f"Accounted for                    : {n_verified+n_failed+n_unfinished+n_invalid} / 120")
print()

if n_invalid:
    raise RuntimeError(
        "INVALID checkpoint(s) detected. Resolve these before Stage 06."
    )

if MODE == "PROGRESS":
    if n_unfinished:
        print("Progress audit complete; production jobs remain")
    else:
        print("All 120 designs are terminal; rerun with TPMS_QC_MODE=FINALIZE")

elif MODE == "FINALIZE":
    if n_unfinished:
        raise RuntimeError(
            f"FINALIZE REFUSED: {n_unfinished} design(s) are still missing/in progress."
        )

    if n_verified + n_failed != 120:
        raise RuntimeError(
            "FINALIZE REFUSED: terminal accounting does not equal 120."
        )

    marker = OUT_ROOT / "05_3_FINALIZED.txt"
    marker.write_text(
        "Stage 05.3 FINALIZED\n"
        f"verified={n_verified}\n"
        f"terminal_qc_failures={n_failed}\n"
        "post_qc_resplit=false\n"
        "targets=von_mises_stress,equivalent_strain\n"
    )

    print(f"Production QC finalized: {n_verified} verified, {n_failed} excluded")

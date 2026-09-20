#!/usr/bin/env python3
"""Consolidate production checkpoints and perform dataset handoff QC.

Converted from `05_3_Production_Consolidation_QC_and_ML_Handoff.ipynb`. Notebook prose and cell output were intentionally omitted.
"""
# ============================================================
# MODULE 0 — IMPORTS + MODE
# ============================================================
import os, json, zipfile, hashlib, re, math, time
from pathlib import Path
from collections import Counter

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

print("Notebook 05.3 mode:", MODE)
print("No FEA will be run in this notebook.")

# ============================================================
# MODULE 1 — ROBUST GOOGLE DRIVE MOUNT
# ============================================================
PROJECT_ROOT = Path(
    os.environ.get("TPMS_PROJECT_ROOT", Path.cwd() / "TPMS_IEEE_BIGDATA")
).expanduser().resolve()
MYDRIVE = PROJECT_ROOT.parent

if not PROJECT_ROOT.exists():
    raise RuntimeError(f"Project root not found: {PROJECT_ROOT}")

OUT_ROOT = PROJECT_ROOT / "Notebook05_3"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("✓ Persistent project root:", PROJECT_ROOT)
print("✓ 05.3 output root:", OUT_ROOT)

# ============================================================
# MODULE 2 — FIND OR UPLOAD AUTHORITATIVE NOTEBOOK-03.1 MANIFEST
# ============================================================
def find_manifest():
    # First search likely project/Drive locations.
    candidates = []
    for root in [PROJECT_ROOT, MYDRIVE, Path.cwd()]:
        try:
            candidates.extend(root.rglob(MANIFEST_NAME))
        except Exception:
            pass

    # Remove duplicates while preserving order.
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
    print("✓ Found authoritative manifest:")
    print(" ", MANIFEST_FILE)
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

print("Manifest:", MANIFEST_FILE)

# ============================================================
# MODULE 3 — AUTHORITATIVE MANIFEST + FROZEN WORKER ASSIGNMENTS
# ============================================================
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

# Preserve authoritative manifest order exactly.
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

print("✓ AUTHORITATIVE MANIFEST PASS")
print("✓ FROZEN SIX-WORKER ASSIGNMENT PASS")
display(pd.DataFrame({
    "worker": list(ASSIGNMENTS),
    "architecture": ["primitive","primitive","diamond","diamond","gyroid","gyroid"],
    "assigned": [len(v) for v in ASSIGNMENTS.values()]
}))
print("\nFrozen split counts:")
display(manifest.groupby(["architecture","iid_split"]).size().unstack(fill_value=0))

# ============================================================
# MODULE 4 — CHECKPOINT ROOTS + SUCCESS/FAILURE VALIDATORS
# ============================================================
CHECKPOINT_ROOTS = {
    worker: PROJECT_ROOT / f"Notebook{worker}" / "production_checkpoints"
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
            if int(meta.get("ccx_returncode", -999)) != 0:
                issues.append(f"ccx_returncode={meta.get('ccx_returncode')}")
            if str(meta.get("mesh_label")) != EXPECTED_MESH_LABEL:
                issues.append(f"mesh_label={meta.get('mesh_label')}")
            if int(meta.get("mapping_k", -1)) != EXPECTED_MAPPING_K:
                issues.append(f"mapping_k={meta.get('mapping_k')}")
            if not math.isclose(float(meta.get("idw_power", np.nan)),
                                EXPECTED_IDW_POWER, rel_tol=0, abs_tol=1e-12):
                issues.append(f"idw_power={meta.get('idw_power')}")
            if meta.get("full_tensor_targets_authorized") is not False:
                issues.append("full_tensor_targets_authorized_not_false")
            if list(meta.get("authorized_targets", [])) != EXPECTED_TARGETS:
                issues.append(f"authorized_targets={meta.get('authorized_targets')}")
            if int(meta.get("graph_components", -1)) != 1:
                issues.append(f"graph_components={meta.get('graph_components')}")
            if int(meta.get("zero_volume_tets", -1)) != 0:
                issues.append(f"zero_volume_tets={meta.get('zero_volume_tets')}")
            radius = float(meta.get("k32_radius_p95_mm", np.nan))
            if not np.isfinite(radius) or radius > MAX_K32_RADIUS_P95_MM:
                issues.append(f"k32_radius_p95_mm={radius}")

    # Make sure compact numerical payloads are actually readable.
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

for worker, root in CHECKPOINT_ROOTS.items():
    print(worker, "->", root, "| exists:", root.exists())

# ============================================================
# MODULE 5 — SCAN ALL 120 ASSIGNED DESIGNS
# ============================================================
manifest_lookup = manifest.set_index("sample_id").to_dict("index")
rows = []

# Detect sample-like folders anywhere in each worker root.
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
                # A folder with intermediate files but no terminal marker is in-progress/partial.
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

print("=" * 88)
print("PRODUCTION STATUS — ALL 120 FROZEN DESIGNS")
print("=" * 88)
display(audit["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count"))

print("\nBy architecture:")
display(pd.crosstab(audit["architecture"], audit["status"]))

print("\nBy frozen iid_split:")
display(pd.crosstab(audit["iid_split"], audit["status"]))

print("\nBy architecture × frozen split (verified only):")
verified = audit[audit.status == "VERIFIED"].copy()
display(verified.groupby(["architecture","iid_split"]).size().unstack(fill_value=0))

# ============================================================
# MODULE 6 — FAILURE PROVENANCE + DISTRIBUTION/QC SUMMARIES
# ============================================================
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
    print("\nINVALID CHECKPOINTS — must be resolved before Notebook 06:")
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

# ============================================================
# MODULE 7 — SPLIT ATTRITION AUDIT
# ============================================================
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

print("=" * 88)
print("FROZEN SPLIT ATTRITION AUDIT")
print("=" * 88)
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

print("IMPORTANT: Notebook 06 must preserve iid_split exactly as shown here.")
print("No post-QC random resplitting is authorized.")

# ============================================================
# MODULE 8 — WRITE CONSOLIDATED 05.3 HANDOFF
# ============================================================
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
    "notebook": "05.3",
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

# ============================================================
# MODULE 9 — FINALIZATION GATE
# ============================================================
n_verified = int((audit.status == "VERIFIED").sum())
n_failed = int((audit.status == "TERMINAL_QC_FAILURE").sum())
n_unfinished = int((audit.status == "MISSING_OR_IN_PROGRESS").sum())
n_invalid = int(audit.status.str.startswith("INVALID", na=False).sum())

print("=" * 88)
print("NOTEBOOK 05.3 CONSOLIDATION GATE")
print("=" * 88)
print(f"Verified successful checkpoints : {n_verified}")
print(f"Terminal predefined QC failures : {n_failed}")
print(f"Missing / still in progress      : {n_unfinished}")
print(f"Invalid/conflicting checkpoints  : {n_invalid}")
print(f"Accounted for                    : {n_verified+n_failed+n_unfinished+n_invalid} / 120")
print()

if n_invalid:
    raise RuntimeError(
        "INVALID checkpoint(s) detected. Resolve these before Notebook 06."
    )

if MODE == "PROGRESS":
    if n_unfinished:
        print("✓ PROGRESS AUDIT COMPLETE")
        print("Remaining production jobs are allowed in PROGRESS mode.")
        print("Rerun Notebook 05.3 after they finish, then change MODE='FINALIZE'.")
    else:
        print("✓ ALL 120 DESIGNS ARE TERMINAL")
        print("Change MODE='FINALIZE' and rerun Module 9 (or Run all) to freeze the handoff.")

elif MODE == "FINALIZE":
    if n_unfinished:
        raise RuntimeError(
            f"FINALIZE REFUSED: {n_unfinished} design(s) are still missing/in progress."
        )

    if n_verified + n_failed != 120:
        raise RuntimeError(
            "FINALIZE REFUSED: terminal accounting does not equal 120."
        )

    # Final handoff marker.
    marker = OUT_ROOT / "05_3_FINALIZED.txt"
    marker.write_text(
        "Notebook 05.3 FINALIZED\n"
        f"verified={n_verified}\n"
        f"terminal_qc_failures={n_failed}\n"
        "post_qc_resplit=false\n"
        "targets=von_mises_stress,equivalent_strain\n"
    )

    print("✓ NOTEBOOK 05.3 FINALIZED")
    print("✓ All 120 frozen designs have a terminal, auditable disposition.")
    print(f"✓ Notebook 06 will receive {n_verified} verified graphs.")
    print(f"✓ {n_failed} predefined QC failure(s) remain excluded with provenance.")
    print("✓ Frozen iid_split assignments are preserved.")
    print("✓ SAFE TO PROCEED TO NOTEBOOK 06")

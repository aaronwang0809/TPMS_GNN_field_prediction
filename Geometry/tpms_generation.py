#!/usr/bin/env python3
"""Generate the deterministic TPMS design set and run geometry QC.

Converted from `03_1_RAM_Safe_Architecture_Batched_TPMS_Dataset_Design_and_Geometry_QC.ipynb`. Notebook prose and cell output were intentionally omitted.
"""
# MODULE 0 — Environment and configuration
from pathlib import Path
import sys, subprocess, importlib.util, json, hashlib, time, math, warnings
import numpy as np
import pandas as pd

REQUIRED = ["numpy", "pandas", "scipy", "skimage", "trimesh", "matplotlib"]
missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
if missing:
    print("Installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scipy", "scikit-image", "trimesh", "matplotlib"])

from scipy.stats import qmc
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

ROOT = Path("/content/tpms_dataset_03") if Path("/content").exists() else Path.cwd() / "tpms_dataset_03"
EXPORT = ROOT / "exports"
PREVIEW = ROOT / "previews"
EXPORT.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)

SEED = 20260916
N_PER_ARCH = 40
ARCHITECTURES = ["gyroid", "diamond", "primitive"]
N_TOTAL = N_PER_ARCH * len(ARCHITECTURES)

DOMAIN_MM = np.array([20.0, 20.0, 20.0])
ORIGIN_MM = np.array([-10.0, -10.0, 0.0])

# Geometry-only screening resolution.
# 96^3 is intentionally used for population screening; production geometry/FEA
# is regenerated later at the frozen 128^3 protocol.
SCREEN_N = 96

# Design ranges
RHO_RANGE = (0.25, 0.55)        # global solid relative density
CELL_RANGE_MM = (4.0, 8.0)      # unit-cell size
GRADE_AMP_RANGE = (0.0, 0.25)   # threshold modulation amplitude
GRADE_MODES = ["uniform", "linear", "sinusoidal"]

# Predeclared geometry QC gates
QC_RHO_ABS_ERR_MAX = 0.010      # screening-resolution absolute density error
QC_COMPONENTS_MAX = 1
QC_MIN_SURFACE_FACES = 2000
QC_MIN_SOLID_VOXELS = 5000

print(f"ROOT: {ROOT}")
print(f"Planned samples: {N_TOTAL} ({N_PER_ARCH} per architecture)")
print(f"Screening grid: {SCREEN_N}^3")
print("NO FEA WILL BE RUN IN NOTEBOOK 03.")

# MODULE 1 — Generate deterministic design manifest

def make_arch_design(arch, n, seed):
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    u = sampler.random(n)

    rho = qmc.scale(u[:, [0]], [RHO_RANGE[0]], [RHO_RANGE[1]]).ravel()
    cell = qmc.scale(u[:, [1]], [CELL_RANGE_MM[0]], [CELL_RANGE_MM[1]]).ravel()
    amp = qmc.scale(u[:, [2]], [GRADE_AMP_RANGE[0]], [GRADE_AMP_RANGE[1]]).ravel()

    # Phase offsets in radians [0, 2pi)
    phase = 2*np.pi*u[:, 3:6]

    # Exactly balanced-ish grading modes within each architecture
    modes = np.array([GRADE_MODES[i % len(GRADE_MODES)] for i in range(n)])
    rng = np.random.default_rng(seed + 991)
    rng.shuffle(modes)

    # Uniform samples have zero grading amplitude by definition.
    amp[modes == "uniform"] = 0.0

    rows = []
    for i in range(n):
        rows.append({
            "architecture": arch,
            "target_relative_density": float(rho[i]),
            "cell_size_mm": float(cell[i]),
            "grading_mode": str(modes[i]),
            "grading_amplitude": float(amp[i]),
            "phase_x_rad": float(phase[i,0]),
            "phase_y_rad": float(phase[i,1]),
            "phase_z_rad": float(phase[i,2]),
        })
    return rows

rows = []
for ai, arch in enumerate(ARCHITECTURES):
    rows += make_arch_design(arch, N_PER_ARCH, SEED + 1000*ai)

design = pd.DataFrame(rows)
design.insert(0, "sample_id", [f"TPMS_{i:04d}" for i in range(len(design))])

assert len(design) == N_TOTAL
assert design["sample_id"].is_unique
assert set(design["architecture"]) == set(ARCHITECTURES)

display(design.head())
print("\nCounts by architecture:")
display(design.groupby("architecture").size().rename("n").to_frame())
print("\nCounts by grading mode:")
display(pd.crosstab(design["architecture"], design["grading_mode"]))

# MODULE 2 — Assign scaffold-level IID and LOAO splits

rng = np.random.default_rng(SEED + 77)
design["iid_split"] = ""

# 40 per architecture -> 24 train, 8 validation, 8 test.
for arch in ARCHITECTURES:
    ids = design.index[design["architecture"] == arch].to_numpy()
    rng.shuffle(ids)
    design.loc[ids[:24], "iid_split"] = "train"
    design.loc[ids[24:32], "iid_split"] = "val"
    design.loc[ids[32:], "iid_split"] = "test"

for heldout in ARCHITECTURES:
    col = f"loao_{heldout}"
    design[col] = np.where(design["architecture"] == heldout, "test", "train_pool")

    # Validation is selected only from the two training architectures:
    pool = design.index[design[col] == "train_pool"].to_numpy()
    # deterministic 20% validation from the 80 non-heldout scaffolds
    local_rng = np.random.default_rng(SEED + 5000 + ARCHITECTURES.index(heldout))
    local_rng.shuffle(pool)
    val_ids = pool[:16]
    design.loc[val_ids, col] = "val"
    design.loc[design[col] == "train_pool", col] = "train"

print("IID split:")
display(pd.crosstab(design["architecture"], design["iid_split"]))

for heldout in ARCHITECTURES:
    print(f"LOAO holdout = {heldout}")
    display(pd.crosstab(design["architecture"], design[f"loao_{heldout}"]))

# MODULE 3 — Implicit TPMS generator

def tpms_field(arch, X, Y, Z, cell_mm, phase):
    k = 2*np.pi / cell_mm
    x = k*X + phase[0]
    y = k*Y + phase[1]
    z = k*Z + phase[2]

    if arch == "gyroid":
        return (np.sin(x)*np.cos(y) +
                np.sin(y)*np.cos(z) +
                np.sin(z)*np.cos(x))

    if arch == "diamond":
        return (np.sin(x)*np.sin(y)*np.sin(z) +
                np.sin(x)*np.cos(y)*np.cos(z) +
                np.cos(x)*np.sin(y)*np.cos(z) +
                np.cos(x)*np.cos(y)*np.sin(z))

    if arch == "primitive":
        return np.cos(x) + np.cos(y) + np.cos(z)

    raise ValueError(f"Unknown architecture: {arch}")


def grading_profile(z_norm, mode):
    if mode == "uniform":
        return np.zeros_like(z_norm)
    if mode == "linear":
        # -1 bottom -> +1 top
        return 2.0*z_norm - 1.0
    if mode == "sinusoidal":
        return np.sin(2*np.pi*z_norm)
    raise ValueError(mode)


def make_grid(n):
    # voxel-center coordinates, preserving 20x20x20 physical envelope
    spacing = DOMAIN_MM / n
    xs = ORIGIN_MM[0] + (np.arange(n)+0.5)*spacing[0]
    ys = ORIGIN_MM[1] + (np.arange(n)+0.5)*spacing[1]
    zs = ORIGIN_MM[2] + (np.arange(n)+0.5)*spacing[2]
    return xs, ys, zs, spacing


def calibrate_mask(row, n=SCREEN_N, tol=2e-4, max_iter=40):
    xs, ys, zs, spacing = make_grid(n)

    # sparse broadcasting avoids allocating three full coordinate grids
    X = xs[:, None, None]
    Y = ys[None, :, None]
    Z = zs[None, None, :]

    phase = np.array([row.phase_x_rad, row.phase_y_rad, row.phase_z_rad])
    F = tpms_field(row.architecture, X, Y, Z, row.cell_size_mm, phase).astype(np.float32)
    absF = np.abs(F)

    z_norm = ((zs - ORIGIN_MM[2]) / DOMAIN_MM[2])[None, None, :]
    g = grading_profile(z_norm, row.grading_mode).astype(np.float32)
    amp = float(row.grading_amplitude)

    target = float(row.target_relative_density)

    # Robust bracket based on field magnitude.
    lo = 0.0
    hi = float(np.quantile(absF, min(0.95, max(0.60, target + 0.30)))) + 1e-6

    def density_at(base):
        threshold = base * np.clip(1.0 + amp*g, 0.20, None)
        return float(np.mean(absF <= threshold))

    while density_at(hi) < target:
        hi *= 1.5
        if hi > float(absF.max())*2 + 1:
            raise RuntimeError("Could not bracket target density")

    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        rho = density_at(mid)
        if abs(rho-target) <= tol:
            break
        if rho < target:
            lo = mid
        else:
            hi = mid

    base = 0.5*(lo+hi)
    threshold = base * np.clip(1.0 + amp*g, 0.20, None)
    mask = absF <= threshold
    rho = float(mask.mean())

    return mask, base, rho, spacing

# MODULE 4 — Geometry QC helpers

def mask_connectivity(mask):
    # 26-neighbor connectivity for solid domain screening
    structure = np.ones((3,3,3), dtype=np.uint8)
    _, ncomp = ndimage.label(mask, structure=structure)
    return int(ncomp)


def boundary_contact(mask):
    return {
        "touch_xmin": bool(mask[0,:,:].any()),
        "touch_xmax": bool(mask[-1,:,:].any()),
        "touch_ymin": bool(mask[:,0,:].any()),
        "touch_ymax": bool(mask[:,-1,:].any()),
        "touch_zmin": bool(mask[:,:,0].any()),
        "touch_zmax": bool(mask[:,:,-1].any()),
    }


def surface_from_mask(mask, spacing):
    # Padding closes cut surfaces and makes watertightness test meaningful.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)

    verts, faces, normals, values = marching_cubes(
        padded,
        level=0.5,
        spacing=tuple(spacing)
    )

    # Convert padded-grid coordinates back to physical coordinates.
    # marching_cubes coordinates are in axis order matching mask: x,y,z.
    verts -= spacing[None, :]
    verts += ORIGIN_MM[None, :]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def mask_sha256(mask):
    return hashlib.sha256(np.packbits(mask.ravel()).tobytes()).hexdigest()


def evaluate_one(row, n=SCREEN_N, build_surface=True):
    t0 = time.time()
    mask, tau, rho, spacing = calibrate_mask(row, n=n)

    solid_vox = int(mask.sum())
    ncomp = mask_connectivity(mask)
    contacts = boundary_contact(mask)

    out = {
        "sample_id": row.sample_id,
        "architecture": row.architecture,
        "target_relative_density": float(row.target_relative_density),
        "screen_relative_density": rho,
        "density_abs_error": abs(rho - float(row.target_relative_density)),
        "calibrated_tau": float(tau),
        "solid_voxels": solid_vox,
        "solid_components_26": ncomp,
        "mask_sha256": mask_sha256(mask),
        **contacts,
    }

    if build_surface:
        mesh = surface_from_mask(mask, spacing)
        out.update({
            "surface_vertices": int(len(mesh.vertices)),
            "surface_faces": int(len(mesh.faces)),
            "surface_watertight": bool(mesh.is_watertight),
            "surface_winding_consistent": bool(mesh.is_winding_consistent),
            "surface_components": int(len(mesh.split(only_watertight=False))),
            "surface_area_mm2": float(mesh.area),
            "surface_volume_mm3": float(abs(mesh.volume)),
        })
        del mesh

    out["runtime_s"] = time.time()-t0

    out["geometry_qc_pass"] = bool(
        out["density_abs_error"] <= QC_RHO_ABS_ERR_MAX
        and out["solid_voxels"] >= QC_MIN_SOLID_VOXELS
        and out["solid_components_26"] <= QC_COMPONENTS_MAX
        and (not build_surface or (
            out["surface_faces"] >= QC_MIN_SURFACE_FACES
            and out["surface_watertight"]
            and out["surface_winding_consistent"]
            and out["surface_components"] == 1
        ))
        and all(contacts.values())
    )

    return out

# MODULE 5 — Fail-fast 3-sample pilot

pilot_rows = [
    design[design["architecture"] == arch].iloc[0]
    for arch in ARCHITECTURES
]

pilot_results = []
for row in pilot_rows:
    print(f"Pilot: {row.sample_id} / {row.architecture}")
    r = evaluate_one(row, SCREEN_N, build_surface=True)
    pilot_results.append(r)
    print(f"  rho={r['screen_relative_density']:.4f}, "
          f"components={r['solid_components_26']}, "
          f"faces={r['surface_faces']:,}, "
          f"watertight={r['surface_watertight']}, "
          f"PASS={r['geometry_qc_pass']}")

pilot_df = pd.DataFrame(pilot_results)
display(pilot_df)

if not pilot_df["geometry_qc_pass"].all():
    raise RuntimeError(
        "PILOT QC FAILED. Stop here; do not run the full 120-sample screen."
    )

print("\n✓ PILOT PASS — safe to proceed to population screening.")

# MODULE 6 — RAM-safe helpers
import gc, ctypes, os

def hard_gc():
    """Collect Python garbage and, on Linux/Colab, ask glibc to return free heap pages."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def run_architecture_batch(arch, checkpoint_name):
    batch = design.loc[design["architecture"] == arch].copy()
    checkpoint = EXPORT / checkpoint_name
    results = []
    t_batch = time.time()

    print("="*72)
    print(f"MODULE 6 BATCH — {arch.upper()} ({len(batch)} scaffolds)")
    print(f"Checkpoint: {checkpoint.name}")
    print("="*72)

    for j, (_, row) in enumerate(batch.iterrows(), start=1):
        print(f"[{j:02d}/{len(batch)}] {row.sample_id} {arch}", end=" ... ")
        r = None
        try:
            r = evaluate_one(row, SCREEN_N, build_surface=True)
            results.append(r)
            print(
                f"rho_err={r['density_abs_error']:.4f} "
                f"comp={r['solid_components_26']} "
                f"faces={r['surface_faces']:,} "
                f"PASS={r['geometry_qc_pass']} "
                f"({r['runtime_s']:.1f}s)"
            )
        except Exception as e:
            results.append({
                "sample_id": row.sample_id,
                "architecture": arch,
                "geometry_qc_pass": False,
                "error": repr(e),
            })
            print("ERROR:", repr(e))
        finally:
            # r contains only scalars/strings, but explicitly drop the local reference.
            del r
            hard_gc()

        # Write after every sample: a runtime crash cannot erase completed work.
        pd.DataFrame(results).to_csv(checkpoint, index=False)

    out = pd.DataFrame(results)
    print(f"\n{arch.capitalize()} batch complete in {(time.time()-t_batch)/60:.1f} min")
    print(f"Passed: {int(out['geometry_qc_pass'].fillna(False).sum())}/{len(out)}")
    print(f"✓ Saved {checkpoint}")
    hard_gc()
    return out

print("✓ RAM-safe Module 6 helpers ready.")

# MODULE 6A — Gyroid only (TPMS_0000–0039)
qc_gyroid = run_architecture_batch(
    "gyroid",
    "03_screen_qc_gyroid.csv"
)

# Keep only the lightweight DataFrame; aggressively release temporary heap pages.
hard_gc()
display(qc_gyroid.tail())

# MODULE 6B — Diamond only (TPMS_0040–0079)
# Run after 6A. The Gyroid checkpoint is already safely on disk.
qc_diamond = run_architecture_batch(
    "diamond",
    "03_screen_qc_diamond.csv"
)

hard_gc()
display(qc_diamond.tail())

# MODULE 6C — Primitive only (TPMS_0080–0119)
# Run after 6B. Earlier architecture checkpoints remain safely on disk.
qc_primitive = run_architecture_batch(
    "primitive",
    "03_screen_qc_primitive.csv"
)

hard_gc()
display(qc_primitive.tail())

# MODULE 6D — Reassemble the authoritative 120-sample QC table
# Read from disk rather than depending on in-memory batch DataFrames.
checkpoint_files = [
    EXPORT / "03_screen_qc_gyroid.csv",
    EXPORT / "03_screen_qc_diamond.csv",
    EXPORT / "03_screen_qc_primitive.csv",
]

missing = [p.name for p in checkpoint_files if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing Module 6 checkpoint(s): " + ", ".join(missing) +
        ". Run the corresponding 6A/6B/6C batch first."
    )

qc = pd.concat([pd.read_csv(p) for p in checkpoint_files], ignore_index=True)

# Restore deterministic design order and enforce one row per planned scaffold.
order = {sid: i for i, sid in enumerate(design["sample_id"])}
qc["_order"] = qc["sample_id"].map(order)
qc = qc.sort_values("_order").drop(columns="_order").reset_index(drop=True)

if len(qc) != N_TOTAL:
    raise RuntimeError(f"Expected {N_TOTAL} QC rows, found {len(qc)}.")
if qc["sample_id"].duplicated().any():
    raise RuntimeError("Duplicate sample_id rows detected in Module 6 checkpoints.")
if set(qc["sample_id"]) != set(design["sample_id"]):
    raise RuntimeError("Checkpoint sample IDs do not exactly match the frozen design.")

qc.to_csv(EXPORT / "03_population_screen_qc_all120.csv", index=False)

print("="*72)
print("MODULE 6 COMPLETE — ARCHITECTURE-BATCHED SCREENING")
print("="*72)
print(f"Rows: {len(qc)}/{N_TOTAL}")
print(f"Passed: {int(qc['geometry_qc_pass'].fillna(False).sum())}/{len(qc)}")
print("Counts by architecture:")
print(qc.groupby("architecture").size())
print("✓ Saved 03_population_screen_qc_all120.csv")
hard_gc()

# MODULE 7 — Merge manifest + QC and inspect failure modes

manifest = design.merge(qc, on=["sample_id", "architecture"], how="left", validate="one_to_one")

print("QC by architecture:")
display(pd.crosstab(manifest["architecture"], manifest["geometry_qc_pass"]))

failed = manifest[~manifest["geometry_qc_pass"].fillna(False)].copy()
if len(failed):
    print(f"\n⚠ {len(failed)} samples failed geometry QC.")
    cols = [c for c in [
        "sample_id","architecture","target_relative_density","cell_size_mm",
        "grading_mode","grading_amplitude","density_abs_error",
        "solid_components_26","surface_watertight","surface_components","error"
    ] if c in failed.columns]
    display(failed[cols])
else:
    print("\n✓ All 120 samples passed geometry QC.")

# Do not silently replace failed designs. Their parameter rows remain in the manifest.
# Any replacement must be generated deterministically in a later explicit repair step.

# MODULE 8 — Dataset diversity / coverage diagnostics

passed = manifest[manifest["geometry_qc_pass"] == True].copy()

summary = passed.groupby("architecture").agg(
    n=("sample_id","count"),
    rho_min=("target_relative_density","min"),
    rho_max=("target_relative_density","max"),
    cell_min_mm=("cell_size_mm","min"),
    cell_max_mm=("cell_size_mm","max"),
    grade_amp_max=("grading_amplitude","max"),
    faces_median=("surface_faces","median"),
    area_median_mm2=("surface_area_mm2","median"),
).reset_index()

display(summary)

# Duplicate geometry hash check at screening resolution
dup_hash = passed[passed.duplicated("mask_sha256", keep=False)].sort_values("mask_sha256")
if len(dup_hash):
    print("⚠ Duplicate screening masks detected:")
    display(dup_hash[["sample_id","architecture","mask_sha256"]])
else:
    print("✓ No duplicate screening masks detected.")

# Simple coverage plots
fig = plt.figure(figsize=(7,5))
for arch in ARCHITECTURES:
    d = passed[passed.architecture == arch]
    plt.scatter(d.cell_size_mm, d.target_relative_density, label=arch, alpha=0.8)
plt.xlabel("Unit-cell size (mm)")
plt.ylabel("Target relative density")
plt.title("Notebook 03 design coverage")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# MODULE 9 — Select representative preview geometries

# Save only 9 preview STLs (3 per architecture) to keep Notebook 03 lightweight.
# Production geometries are regenerated deterministically later from the manifest.

preview_ids = []

for arch in ARCHITECTURES:
    d = passed[passed.architecture == arch].sort_values("target_relative_density")
    if len(d) >= 3:
        pick = d.iloc[[0, len(d)//2, -1]]
    else:
        pick = d
    preview_ids += pick.sample_id.tolist()

for sid in preview_ids:
    row = design.loc[design.sample_id == sid].iloc[0]
    mask, tau, rho, spacing = calibrate_mask(row, SCREEN_N)
    mesh = surface_from_mask(mask, spacing)
    out = PREVIEW / f"{sid}_{row.architecture}.stl"
    mesh.export(out)
    print(f"✓ {out.name} ({len(mesh.faces):,} faces)")

print(f"\nSaved {len(preview_ids)} preview STLs only.")

# MODULE 10 — Final dataset-design QC and exports

EXPECTED_IID = {
    "train": 72,
    "val": 24,
    "test": 24,
}

iid_counts = design["iid_split"].value_counts().to_dict()

DESIGN_COUNT_PASS = len(design) == 120
ARCH_BALANCE_PASS = all((design.architecture == a).sum() == 40 for a in ARCHITECTURES)
IID_SPLIT_PASS = all(iid_counts.get(k,0) == v for k,v in EXPECTED_IID.items())
UNIQUE_PARAMETER_ROWS_PASS = (
    len(design.drop_duplicates([
        "architecture","target_relative_density","cell_size_mm","grading_mode",
        "grading_amplitude","phase_x_rad","phase_y_rad","phase_z_rad"
    ])) == len(design)
)
ALL_GEOMETRY_QC_PASS = bool(manifest["geometry_qc_pass"].fillna(False).all())
NO_DUPLICATE_MASKS_PASS = not manifest[
    manifest["geometry_qc_pass"] == True
].duplicated("mask_sha256").any()

final_qc = pd.DataFrame([
    ["120 planned independent scaffolds", DESIGN_COUNT_PASS],
    ["40 samples per TPMS architecture", ARCH_BALANCE_PASS],
    ["IID split = 72/24/24", IID_SPLIT_PASS],
    ["unique parameter rows", UNIQUE_PARAMETER_ROWS_PASS],
    ["all geometries pass screening QC", ALL_GEOMETRY_QC_PASS],
    ["no duplicate screening masks", NO_DUPLICATE_MASKS_PASS],
], columns=["gate","pass"])

display(final_qc)

READY_FOR_PRODUCTION_BATCH = bool(final_qc["pass"].all())

# Save complete manifest regardless of pass/fail so failures are auditable.
manifest.to_csv(EXPORT / "03_tpms_dataset_manifest_with_qc.csv", index=False)
design.to_csv(EXPORT / "03_frozen_parameter_design.csv", index=False)
final_qc.to_csv(EXPORT / "03_final_qc.csv", index=False)

config = {
    "seed": SEED,
    "n_total": N_TOTAL,
    "n_per_architecture": N_PER_ARCH,
    "architectures": ARCHITECTURES,
    "domain_mm": DOMAIN_MM.tolist(),
    "origin_mm": ORIGIN_MM.tolist(),
    "screen_resolution": SCREEN_N,
    "production_resolution_required": 128,
    "relative_density_range": list(RHO_RANGE),
    "cell_size_mm_range": list(CELL_RANGE_MM),
    "grading_amplitude_range": list(GRADE_AMP_RANGE),
    "grading_modes": GRADE_MODES,
    "iid_split": EXPECTED_IID,
    "loao_protocol": "one complete TPMS architecture held out for test; 16/80 remaining scaffolds used for validation",
    "geometry_qc": {
        "density_abs_error_max": QC_RHO_ABS_ERR_MAX,
        "solid_components_26_max": QC_COMPONENTS_MAX,
        "min_surface_faces": QC_MIN_SURFACE_FACES,
        "min_solid_voxels": QC_MIN_SOLID_VOXELS,
        "requires_watertight": True,
        "requires_winding_consistent": True,
        "requires_all_six_domain_face_contacts": True,
    },
    "ready_for_production_batch": READY_FOR_PRODUCTION_BATCH,
}

with open(EXPORT / "03_dataset_design_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "="*72)
if READY_FOR_PRODUCTION_BATCH:
    print("✓ NOTEBOOK 03 COMPLETE")
    print("✓ READY_FOR_PRODUCTION_BATCH = True")
    print("The 120-sample parameter design is frozen.")
else:
    print("⚠ NOTEBOOK 03 COMPLETE WITH QC FAILURES")
    print("READY_FOR_PRODUCTION_BATCH = False")
    print("Do NOT start batch FEA until failed geometry rows are reviewed.")
print("="*72)

print("\nExports:")
for p in sorted(EXPORT.glob("*")):
    print(" ", p.name)

# MODULE 11 — Download Notebook 03 outputs (Colab only)

# Run this only after Module 10.
import zipfile

zip_path = ROOT / "03_TPMS_DATASET_DESIGN_OUTPUTS.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in EXPORT.glob("*"):
        if p.is_file():
            zf.write(p, arcname=p.name)
    for p in PREVIEW.glob("*"):
        if p.is_file():
            zf.write(p, arcname=f"previews/{p.name}")

print(f"Created: {zip_path}")
print(f"Size: {zip_path.stat().st_size/1024**2:.2f} MB")

try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    print("Not running in Colab; download manually from:", zip_path)

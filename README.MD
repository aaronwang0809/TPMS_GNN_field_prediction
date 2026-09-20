# TPMS_GNN_field_prediction

Code accompanying the paper **“Topology-Aware Graph Neural Networks for Stress-Strain Mapping in Generative 3D Porous Titanium Scaffolds.”**

The repository contains the code for TPMS generation, canonical graph construction, finite-element analysis and field mapping, dataset assembly, and graph neural network training and evaluation.

Run the production pipeline in this order: `Geometry/tpms_generation.py` → prepare the approved Stage 03.1 handoff archive `03_1_FINAL_TPMS_DATASET_DESIGN.zip`, containing `03_FINAL_population_120_WITH_SPLITS.csv` → `FEA/fea_validation.py` → `FEA/production_fea.py` for each worker from `5.2A` through `5.2F` → `Dataset/production_qc.py` in `PROGRESS` mode and then `FINALIZE` mode → `Dataset/dataset_assembly.py` → `GNN/train.py` → `GNN/maurizi_benchmark.py`; `GNN/models.py` is imported by the training scripts and is not run directly.

`Geometry/graph_construction.py` and `FEA/fea_field_mapping.py` are pre-production validation/reference scripts rather than required steps in the production dataset pipeline; the field-mapping script separately requires `02_8_inputs.zip` containing `balanced_mesh.npz`, `canonical_structural_graph_128.pt`, and `02_7_final_convergence_bookkeeping.json`.

Before running the pipeline, install Python 3.11, NumPy, pandas, SciPy, scikit-image, trimesh, Matplotlib, NetworkX, TetGen, psutil, a CUDA-compatible build of PyTorch, PyTorch Geometric, and CalculiX with the `ccx` executable available on the system `PATH`.

## Citation

A. Wang, “Topology-Aware Graph Neural Networks for Stress-Strain Mapping in Generative 3D Porous Titanium Scaffolds,” 2026.

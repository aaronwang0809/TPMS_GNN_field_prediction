# TPMS_GNN_field_prediction

Code accompanying the paper **“Topology-Aware Graph Neural Networks for Stress-Strain Mapping in Generative 3D Porous Titanium Scaffolds.”**

The repository contains the code for TPMS generation, canonical graph construction, finite-element analysis and field mapping, dataset assembly, and graph neural network training and evaluation.

Run the files in this order: `Geometry/graph_construction.py` → `FEA/fea_field_mapping.py` → `Geometry/tpms_generation.py` → `FEA/fea_validation.py` → `FEA/production_fea.py` → `Dataset/production_qc.py` → `Dataset/dataset_assembly.py` → `GNN/train.py` → `GNN/maurizi_benchmark.py`.

## Citation

A. Wang, “Topology-Aware Graph Neural Networks for Stress-Strain Mapping in Generative 3D Porous Titanium Scaffolds,” 2026.

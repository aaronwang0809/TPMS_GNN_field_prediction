#!/usr/bin/env python3
"""Run full-field FEA and map element fields onto a canonical graph.

Converted from `02_8_CORRECTED_FEA_to_Canonical_Graph_Ground_Truth_Mapping.ipynb`. Notebook prose and cell output were intentionally omitted.
"""
# MODULE 0 — Install
# Notebook-only command removed: !apt-get update -qq
# Notebook-only command removed: !apt-get install -y -qq calculix-ccx
# Notebook-only command removed: %pip install -q numpy pandas scipy matplotlib psutil torch-geometric
print("✓ Dependencies installed")

# MODULE 1 — Imports, directories, frozen protocol
import os,re,gc,json,time,shutil,hashlib,subprocess,zipfile,sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, psutil
from scipy.spatial import cKDTree
import torch

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

SEED=42
np.random.seed(SEED)

ROOT=Path.cwd()/"tpms_fea_2_8"
INPUT=ROOT/"input"; RUN=ROOT/"production_run"; EXPORT=ROOT/"exports"; FIG=ROOT/"figures"
for p in [INPUT,RUN,EXPORT,FIG]: p.mkdir(parents=True,exist_ok=True)

E_MPA=110000.0
NU=0.33
TARGET_MACRO_STRAIN=0.01
MAPPING_K_VALUES=[16,32,64]
PRIMARY_K=32
IDW_POWER=2.0
NORMAL_Z_MIN=0.80

CCX=shutil.which("ccx")
assert CCX is not None, "CalculiX ccx not found."
print("CalculiX:",CCX)
print("RAM available:",round(psutil.virtual_memory().available/1e9,2),"GB")

# MODULE 2 — Upload and unpack the standalone handoff bundle
try:
    from google.colab import files
    uploaded=files.upload()
    candidates=[Path(x) for x in uploaded if x.endswith(".zip")]
    if not candidates:
        raise RuntimeError("Upload 02_8_inputs.zip.")
    bundle=candidates[0]
except ImportError:
    candidates=list(Path.cwd().glob("02_8_inputs.zip"))+list(INPUT.glob("02_8_inputs.zip"))
    if not candidates:
        raise RuntimeError("Place 02_8_inputs.zip in the working directory.")
    bundle=candidates[0]

with zipfile.ZipFile(bundle,"r") as z:
    z.extractall(INPUT)

print("Bundle contents:")
for p in sorted(INPUT.rglob("*")):
    if p.is_file():
        print(" ",p.relative_to(INPUT),f"{p.stat().st_size/1e6:.2f} MB")

# MODULE 3 — Load and validate frozen FE mesh + actual Notebook-01.2 PyG graph
mesh_file=next(iter(INPUT.rglob("balanced_mesh.npz")),None)
graph_file=next(iter(INPUT.rglob("canonical_structural_graph_128.pt")),None)
book_file=next(iter(INPUT.rglob("02_7_final_convergence_bookkeeping.json")),None)

if mesh_file is None:
    raise RuntimeError("balanced_mesh.npz missing from bundle.")
if graph_file is None:
    raise RuntimeError("canonical_structural_graph_128.pt missing from bundle.")
if book_file is None:
    raise RuntimeError("02_7_final_convergence_bookkeeping.json missing from bundle.")

m=np.load(mesh_file)
P=np.asarray(m["points"],np.float64)
T=np.asarray(m["tets"],np.int64)

# Notebook 01.2 saved a torch_geometric.data.Data object.
try:
    struct_data=torch.load(graph_file,map_location="cpu",weights_only=False)
except TypeError:
    struct_data=torch.load(graph_file,map_location="cpu")

required_graph_attrs=["x","pos","edge_index","edge_attr"]
missing=[a for a in required_graph_attrs if not hasattr(struct_data,a)]
if missing:
    raise RuntimeError("Frozen PyG graph missing attributes: "+", ".join(missing))

GX=struct_data.x.detach().cpu().numpy().astype(np.float32,copy=False)
GPOS=struct_data.pos.detach().cpu().numpy().astype(np.float64,copy=False)
EDGE_INDEX=struct_data.edge_index.detach().cpu().numpy().astype(np.int64,copy=False)
DIRECTED_EDGE_ATTR=struct_data.edge_attr.detach().cpu().numpy().astype(np.float32,copy=False)

# 01.2 stores 612 directed PyG edges = two directions for 306 structural edges.
# Freeze one canonical undirected copy for NPZ export while preserving the original PyG object.
src,dst=EDGE_INDEX
keep=src<dst
GEDGES=np.stack([src[keep],dst[keep]],axis=1)
GEDGE_ATTR=DIRECTED_EDGE_ATTR[keep]

book=json.loads(book_file.read_text())

print("FE mesh:",P.shape,T.shape)
print("Frozen PyG graph:",struct_data)
print("Canonical positions:",GPOS.shape)
print("Directed PyG edges:",EDGE_INDEX.shape[1])
print("Undirected canonical edges:",GEDGES.shape)
print("Edge attributes:",GEDGE_ATTR.shape)
print("Bookkeeping stress-field pass:",book.get("stress_field_convergence_pass"))
print("Bookkeeping production tets:",book.get("production_tets"))

assert len(T)==1_901_495, f"Expected frozen balanced mesh (1,901,495 tets), got {len(T):,}"
assert GX.shape==(197,8), f"Expected frozen x=[197,8], got {GX.shape}"
assert GPOS.shape==(197,3), f"Expected frozen pos=[197,3], got {GPOS.shape}"
assert EDGE_INDEX.shape==(2,612), f"Expected frozen edge_index=[2,612], got {EDGE_INDEX.shape}"
assert DIRECTED_EDGE_ATTR.shape==(612,8), f"Expected frozen edge_attr=[612,8], got {DIRECTED_EDGE_ATTR.shape}"
assert GEDGES.shape==(306,2), f"Expected 306 undirected edges, got {GEDGES.shape}"
assert book.get("stress_field_convergence_pass") is True
assert book.get("production_mesh")=="balanced"
assert int(book.get("production_tets"))==len(T)

print("✓ Actual Notebook-01.2 PyG graph + frozen FE input QC PASS")

# MODULE 4 — True exterior-facet BCs and deck helpers
def exterior_faces(t):
    ff=np.vstack([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[1,2,3]],t[:,[2,0,3]]])
    ss=np.sort(ff,axis=1)
    order=np.lexsort((ss[:,2],ss[:,1],ss[:,0]))
    s=ss[order]
    same=np.all(s[1:]==s[:-1],axis=1)
    starts=np.r_[0,np.where(~same)[0]+1]
    ends=np.r_[starts[1:],len(s)]
    return ff[order[starts[(ends-starts)==1]]]

# Reconstruct the accepted 128-grid pitch from FE specimen extent.
# Frozen geometry spans approximately 20 mm; the earlier protocol used pitch=height/128.
height=float(P[:,2].max()-P[:,2].min())
pitch=height/128.0
TOP_DISPLACEMENT_MM=-TARGET_MACRO_STRAIN*height

bf=exterior_faces(T)
tri=P[bf]
fc=tri.mean(1)
nv=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
nz=np.abs((nv/np.maximum(np.linalg.norm(nv,axis=1)[:,None],1e-15))[:,2])
z0,z1=P[:,2].min(),P[:,2].max()
tol=1.5*pitch
bottom=np.unique(bf[(fc[:,2]<=z0+tol)&(nz>=NORMAL_Z_MIN)])
top=np.unique(bf[(fc[:,2]>=z1-tol)&(nz>=NORMAL_Z_MIN)])

xy=P[bottom,:2]; cen=xy.mean(0)
anchor_a=int(bottom[np.argmin(np.linalg.norm(xy-cen,axis=1))])
anchor_b=int(bottom[np.argmax(np.linalg.norm(P[bottom,:2]-P[anchor_a,:2],axis=1))])

def write_ids(f,ids,n=16,one_based=True):
    z=np.asarray(ids,dtype=np.int64)+(1 if one_based else 0)
    for i in range(0,len(z),n):
        f.write(",".join(map(str,z[i:i+n]))+"\n")

print("height:",height,"mm")
print("top displacement:",TOP_DISPLACEMENT_MM,"mm")
print("bottom nodes:",len(bottom),"top nodes:",len(top))
assert len(bottom)>1000 and len(top)>1000
print("✓ Physical exterior-facet BC reconstruction PASS")

# MODULE 5 — Write production full-field CalculiX deck
deck=RUN/"model.inp"
with open(deck,"w",buffering=1024*1024) as f:
    f.write("*HEADING\nTPMS production full-field solve for graph mapping\n*NODE\n")
    for i,x in enumerate(P,1):
        f.write(f"{i},{x[0]:.8g},{x[1]:.8g},{x[2]:.8g}\n")
    f.write("*ELEMENT,TYPE=C3D4,ELSET=SOLID\n")
    for i,x in enumerate(T,1):
        y=x+1
        f.write(f"{i},{y[0]},{y[1]},{y[2]},{y[3]}\n")
    for nm,arr in [("BOTTOM",bottom),("TOP",top),("ANCHOR_A",[anchor_a]),("ANCHOR_B",[anchor_b])]:
        f.write(f"*NSET,NSET={nm}\n"); write_ids(f,arr)
    f.write("*MATERIAL,NAME=TI64\n*ELASTIC\n")
    f.write(f"{E_MPA},{NU}\n")
    f.write("*SOLID SECTION,ELSET=SOLID,MATERIAL=TI64\n")
    f.write("*STEP\n*STATIC\n*BOUNDARY\n")
    f.write("BOTTOM,3,3,0.\n")
    f.write(f"TOP,3,3,{TOP_DISPLACEMENT_MM}\n")
    f.write("ANCHOR_A,1,2,0.\nANCHOR_B,2,2,0.\n")
    f.write("*NODE PRINT,NSET=TOP\nRF\n")
    f.write("*EL PRINT,ELSET=SOLID\nS,E\n")
    f.write("*END STEP\n")

first=deck.read_text(errors="strict").splitlines()[:4]
assert first[0]=="*HEADING" and first[2]=="*NODE"
print("Deck sanity PASS:",first)
print("Deck size:",round(deck.stat().st_size/1e6,1),"MB")
print("RAM available:",round(psutil.virtual_memory().available/1e9,2),"GB")

# MODULE 6 — Run production FEA exactly once
dat=RUN/"model.dat"
if dat.exists() and dat.stat().st_size>1_000_000:
    print("Existing production model.dat found; solve is NOT rerun.")
else:
    print("Starting production balanced full-field CalculiX solve...")
    t0=time.time()
    cp=subprocess.run([CCX,"model"],cwd=RUN,capture_output=True,text=True)
    solve_s=time.time()-t0
    print("return code:",cp.returncode,"solve minutes:",round(solve_s/60,2))
    if cp.returncode!=0:
        print(cp.stdout[-5000:]); print(cp.stderr[-5000:])
        raise RuntimeError("Production full-field CalculiX solve failed.")
    print("✓ Production solve completed")

print("DAT:",round(dat.stat().st_size/1e6,1),"MB")

# MODULE 7 — Inspect actual DAT headings before parsing
heads=[]
with open(dat,"r",errors="ignore") as f:
    for line in f:
        lo=line.lower()
        if ("stresses (" in lo or "strains (" in lo or "forces (" in lo):
            heads.append(line.strip())
            if len(heads)>=10: break
print("\n".join(heads))
if not any("stresses" in h.lower() for h in heads):
    raise RuntimeError("Stress section not found.")
if not any("strains" in h.lower() for h in heads):
    raise RuntimeError("Strain section not found. STOP before mapping.")
print("✓ Stress + strain sections detected")

# MODULE 8 — Streaming full-field parser (float32, element ordered)
N=len(T)
S=np.full((N,6),np.nan,dtype=np.float32)
Eps=np.full((N,6),np.nan,dtype=np.float32)
rf_sum=np.zeros(3,dtype=np.float64)
rf_count=0
mode=None
stress_count=strain_count=0
top_ids=set(map(int,top+1))

with open(dat,"r",errors="ignore") as f:
    for line in f:
        lo=line.lower()
        if "forces (fx,fy,fz) for set top" in lo:
            mode="rf"; continue
        if "stresses (elem, integ.pnt." in lo and "for set solid" in lo:
            mode="stress"; continue
        if "strains (elem, integ.pnt." in lo and "for set solid" in lo:
            mode="strain"; continue
        if not line.strip(): continue
        p=line.split()
        if mode=="rf" and len(p)==4:
            try:
                node=int(p[0])
                if node in top_ids:
                    rf_sum += np.asarray(p[1:4],float); rf_count+=1
            except ValueError: pass
        elif mode in ("stress","strain") and len(p)==8:
            try:
                eid=int(p[0])-1
                vals=np.asarray(p[2:8],np.float32)
                if 0<=eid<N:
                    if mode=="stress": S[eid]=vals; stress_count+=1
                    else: Eps[eid]=vals; strain_count+=1
            except ValueError: pass

validS=np.all(np.isfinite(S),axis=1)
validE=np.all(np.isfinite(Eps),axis=1)
print("reaction rows:",rf_count,"reaction N:",rf_sum)
print("stress valid:",validS.sum(),"/",N)
print("strain valid:",validE.sum(),"/",N)
if validS.mean()<0.999 or validE.mean()<0.999:
    raise RuntimeError("Full-field parser did not recover >=99.9% of elements.")
print("✓ Full stress/strain field parsed")

# MODULE 9 — Derived FE invariants + constitutive consistency QC
def von_mises(A):
    a,b,c,d,e,f=[A[:,i].astype(np.float64) for i in range(6)]
    return np.sqrt(.5*((a-b)**2+(b-c)**2+(c-a)**2)+3*(d*d+e*e+f*f))

VM=von_mises(S).astype(np.float32)

# CalculiX component order from the printed tensor is:
# xx, yy, zz, xy, xz, yz.
# Equivalent strain uses engineering shear strains for printed E.
ex,ey,ez,gxy,gxz,gyz=[Eps[:,i].astype(np.float64) for i in range(6)]
# deviatoric normal strain + tensor shear = engineering shear/2
m=(ex+ey+ez)/3
EQ=np.sqrt((2/3)*((ex-m)**2+(ey-m)**2+(ez-m)**2 +
                  2*((gxy/2)**2+(gxz/2)**2+(gyz/2)**2))).astype(np.float32)

print("VM MPa median/P95/P99:",
      np.nanmedian(VM),np.nanquantile(VM,.95),np.nanquantile(VM,.99))
print("Equivalent strain median/P95/P99:",
      np.nanmedian(EQ),np.nanquantile(EQ,.95),np.nanquantile(EQ,.99))
print("Reaction Fz N:",rf_sum[2])

# MODULE 10 — FE centroid tree and graph-to-FE distance QC
CENT=(P[T[:,0]]+P[T[:,1]]+P[T[:,2]]+P[T[:,3]])/4.0
tree=cKDTree(CENT)

dist64,idx64=tree.query(GPOS,k=max(MAPPING_K_VALUES),workers=-1)
if dist64.ndim==1:
    dist64=dist64[:,None]; idx64=idx64[:,None]

distance_qc=pd.DataFrame({
    "nearest_mm":dist64[:,0],
    "k32_farthest_mm":dist64[:,31],
    "k64_farthest_mm":dist64[:,63],
})
display(distance_qc.describe(percentiles=[.5,.9,.95,.99]))

# A graph node should lie close to the solid FE domain.
nearest_p95=float(np.quantile(dist64[:,0],.95))
nearest_max=float(np.max(dist64[:,0]))
DIST_GATE=max(2*pitch,0.35)  # predeclared geometry-scale sanity gate
DISTANCE_QC_PASS=nearest_p95<=DIST_GATE

print("nearest-distance P95:",nearest_p95,"mm")
print("nearest-distance max:",nearest_max,"mm")
print("distance gate:",DIST_GATE,"mm")
print("DISTANCE_QC_PASS =",DISTANCE_QC_PASS)

# MODULE 11 — IDW mapping for k = 16, 32, 64
def idw_map(values,dist,idx,k,power=2.0,eps=1e-9):
    d=dist[:,:k]
    ii=idx[:,:k]
    w=1.0/np.maximum(d,eps)**power
    w=w/w.sum(axis=1,keepdims=True)
    if values.ndim==1:
        return np.sum(w*values[ii],axis=1)
    return np.sum(w[:,:,None]*values[ii],axis=1)

mapped={}
for k in MAPPING_K_VALUES:
    mapped[k]={
        "stress":idw_map(S,dist64,idx64,k).astype(np.float32),
        "strain":idw_map(Eps,dist64,idx64,k).astype(np.float32),
        "vm":idw_map(VM,dist64,idx64,k).astype(np.float32),
        "eq_strain":idw_map(EQ,dist64,idx64,k).astype(np.float32),
    }

print("Mapped nodes:",len(GPOS),"for k =",MAPPING_K_VALUES)

# MODULE 13 — Freeze primary k=32 graph targets and save tabular/PyG outputs
Ystress=mapped[PRIMARY_K]["stress"]
Ystrain=mapped[PRIMARY_K]["strain"]
Yvm=mapped[PRIMARY_K]["vm"]
Yeq=mapped[PRIMARY_K]["eq_strain"]

colsS=["sxx","syy","szz","sxy","sxz","syz"]
colsE=["exx","eyy","ezz","exy","exz","eyz"]

node_df=pd.DataFrame({
    "node_id":np.arange(len(GPOS)),
    "x":GPOS[:,0],"y":GPOS[:,1],"z":GPOS[:,2],
    "nearest_fe_centroid_mm":dist64[:,0],
    "von_mises_MPa":Yvm,
    "equivalent_strain":Yeq,
})
for j,c in enumerate(colsS):
    node_df[c+"_MPa"]=Ystress[:,j]
for j,c in enumerate(colsE):
    node_df[c]=Ystrain[:,j]

node_df.to_csv(EXPORT/"canonical_graph_node_targets_k32.csv",index=False)
sens.to_csv(EXPORT/"mapping_k_sensitivity.csv",index=False)
distance_qc.to_csv(EXPORT/"mapping_distance_qc.csv",index=False)

target_feature_order=[
    "sxx_MPa","syy_MPa","szz_MPa","sxy_MPa","sxz_MPa","syz_MPa",
    "exx","eyy","ezz","exy","exz","eyz","von_mises_MPa","equivalent_strain"
]

# Portable NumPy export.
np.savez_compressed(
    EXPORT/"canonical_graph_with_fea_targets.npz",
    x=GX,
    pos=GPOS,
    edge_index=EDGE_INDEX,
    edge_attr=DIRECTED_EDGE_ATTR,
    undirected_edges=GEDGES,
    undirected_edge_attr=GEDGE_ATTR,
    y_stress=Ystress,
    y_strain=Ystrain,
    y_vm=Yvm,
    y_eq_strain=Yeq,
    target_feature_order=np.asarray(target_feature_order),
    mapping_k=np.int64(PRIMARY_K)
)

# Preserve the exact frozen Notebook-01.2 PyG graph and append FEA labels.
graph_with_targets=struct_data.clone()
graph_with_targets.y_stress=torch.as_tensor(Ystress,dtype=torch.float32)
graph_with_targets.y_strain=torch.as_tensor(Ystrain,dtype=torch.float32)
graph_with_targets.y_vm=torch.as_tensor(Yvm[:,None],dtype=torch.float32)
graph_with_targets.y_eq_strain=torch.as_tensor(Yeq[:,None],dtype=torch.float32)
graph_with_targets.mapping_k=PRIMARY_K
graph_with_targets.target_feature_order=target_feature_order

torch.save(
    graph_with_targets,
    EXPORT/"canonical_structural_graph_128_with_fea_targets.pt"
)

print(node_df.head())
print("✓ Primary graph-target NPZ + PyG exports written")

# MODULE 13 — Freeze primary k=32 graph targets and save tabular outputs
Ystress=mapped[PRIMARY_K]["stress"]
Ystrain=mapped[PRIMARY_K]["strain"]
Yvm=mapped[PRIMARY_K]["vm"]
Yeq=mapped[PRIMARY_K]["eq_strain"]

colsS=["sxx","syy","szz","sxy","sxz","syz"]
colsE=["exx","eyy","ezz","exy","exz","eyz"]

node_df=pd.DataFrame({
    "node_id":np.arange(len(GPOS)),
    "x":GPOS[:,0],"y":GPOS[:,1],"z":GPOS[:,2],
    "nearest_fe_centroid_mm":dist64[:,0],
    "von_mises_MPa":Yvm,
    "equivalent_strain":Yeq,
})
for j,c in enumerate(colsS): node_df[c+"_MPa"]=Ystress[:,j]
for j,c in enumerate(colsE): node_df[c]=Ystrain[:,j]

node_df.to_csv(EXPORT/"canonical_graph_node_targets_k32.csv",index=False)
sens.to_csv(EXPORT/"mapping_k_sensitivity.csv",index=False)
distance_qc.to_csv(EXPORT/"mapping_distance_qc.csv",index=False)

np.savez_compressed(
    EXPORT/"canonical_graph_with_fea_targets.npz",
    x=GX,pos=GPOS,edges=GEDGES,edge_attr=GEDGE_ATTR,
    y_stress=Ystress,y_strain=Ystrain,y_vm=Yvm,y_eq_strain=Yeq,
    target_feature_order=np.asarray(
        ["sxx_MPa","syy_MPa","szz_MPa","sxy_MPa","sxz_MPa","syz_MPa",
         "exx","eyy","ezz","exy","exz","eyz","von_mises_MPa","equivalent_strain"]
    ),
    mapping_k=np.int64(PRIMARY_K)
)
print(node_df.head())
print("✓ Primary graph-target exports written")

# MODULE 14 — Reproducible figures
fig=plt.figure(figsize=(8,7))
ax=fig.add_subplot(111,projection="3d")
sc=ax.scatter(GPOS[:,0],GPOS[:,1],GPOS[:,2],c=Yvm,s=24)
fig.colorbar(sc,ax=ax,label="Mapped von Mises stress [MPa]")
ax.set_title("Canonical structural graph — mapped production FEA")
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
plt.tight_layout()
plt.savefig(FIG/"canonical_graph_mapped_vm.png",dpi=220,bbox_inches="tight")
plt.show()

plt.figure(figsize=(6.5,4.5))
for k in MAPPING_K_VALUES:
    plt.plot(np.sort(mapped[k]["vm"]),label=f"k={k}")
plt.xlabel("Canonical node rank")
plt.ylabel("Mapped von Mises stress [MPa]")
plt.title("FE→graph mapping sensitivity")
plt.legend()
plt.tight_layout()
plt.savefig(FIG/"mapping_k_sensitivity_vm.png",dpi=220,bbox_inches="tight")
plt.show()

# MODULE 15 — Final QC and manifest
FINITE_TARGETS_PASS=(
    np.all(np.isfinite(Ystress)) and np.all(np.isfinite(Ystrain)) and
    np.all(np.isfinite(Yvm)) and np.all(np.isfinite(Yeq))
)
GRAPH_IDENTITY_PASS=(len(GPOS)==197 and len(GEDGES)==306)
FULL_FIELD_PASS=(validS.mean()>=.999 and validE.mean()>=.999)

READY_FOR_ML=bool(
    book.get("stress_field_convergence_pass") and
    FULL_FIELD_PASS and GRAPH_IDENTITY_PASS and FINITE_TARGETS_PASS and
    DISTANCE_QC_PASS and MAPPING_SENSITIVITY_PASS
)

qc=pd.DataFrame([
    ["02.7 stress-field convergence",bool(book.get("stress_field_convergence_pass"))],
    ["full FE stress/strain parsed",FULL_FIELD_PASS],
    ["canonical graph identity",GRAPH_IDENTITY_PASS],
    ["finite mapped targets",FINITE_TARGETS_PASS],
    ["FE→graph distance QC",DISTANCE_QC_PASS],
    ["k32→k64 mapping sensitivity",MAPPING_SENSITIVITY_PASS],
],columns=["gate","pass"])
display(qc)

manifest={
    "notebook":"02_8_FEA_to_Canonical_Graph_Ground_Truth_Mapping",
    "production_mesh":"balanced",
    "production_nodes":int(len(P)),
    "production_tets":int(len(T)),
    "material":{"E_MPa":E_MPA,"nu":NU},
    "macro_strain":TARGET_MACRO_STRAIN,
    "linear_response_note":"1% nominal compression used as normalized linear-elastic response; stresses/reactions scale linearly.",
    "canonical_graph_nodes":int(len(GPOS)),
    "canonical_graph_edges":int(len(GEDGES)),
    "mapping":{"method":"k-nearest FE-centroid inverse-distance weighting",
               "k_sensitivity":MAPPING_K_VALUES,"primary_k":PRIMARY_K,"power":IDW_POWER},
    "mapping_sensitivity":sens.to_dict(orient="records"),
    "distance_qc":{"nearest_p95_mm":nearest_p95,"nearest_max_mm":nearest_max,"gate_mm":DIST_GATE},
    "qc":{row.gate:bool(row["pass"]) for _,row in qc.iterrows()},
    "ready_for_ml":READY_FOR_ML
}
(EXPORT/"02_8_manifest.json").write_text(json.dumps(manifest,indent=2))
qc.to_csv(EXPORT/"02_8_qc_gates.csv",index=False)

print("="*70)
print("READY_FOR_ML =",READY_FOR_ML)
if READY_FOR_ML:
    print("✓ Notebook 02.8 COMPLETE — canonical graph targets frozen.")
    print("✓ NEXT: Notebook 03 — generative multi-scaffold dataset production.")
else:
    print("STOP: one or more mapping/full-field QC gates failed.")
print("="*70)

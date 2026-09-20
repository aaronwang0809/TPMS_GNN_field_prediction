#!/usr/bin/env python3
"""Run resumable production FEA for one of the six frozen worker shards.

"""

import os, sys, re, gc, json, time, shutil, subprocess, ctypes, hashlib, zipfile
from pathlib import Path

WORKER_REQUEST = os.environ.get("TPMS_WORKER", "5.2A").upper()

def ensure_package(import_name, pip_name=None):
    import importlib.util
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )

for imp, pipn in [
    ("numpy","numpy"), ("pandas","pandas"), ("scipy","scipy"),
    ("skimage","scikit-image"), ("trimesh","trimesh"),
    ("tetgen","tetgen"), ("psutil","psutil")
]:
    ensure_package(imp,pipn)

import numpy as np
import pandas as pd
import psutil
import trimesh
import tetgen
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize

CCX = shutil.which("ccx")
if CCX is None:
    raise RuntimeError(
        "CalculiX executable 'ccx' is required on PATH. Install CalculiX "
        "before launching production FEA."
    )
CCX_PATH = CCX

default_project_root = Path(
    os.environ.get("TPMS_PROJECT_ROOT", Path.cwd() / "TPMS_IEEE_BIGDATA")
).expanduser().resolve()
default_worker_root = default_project_root / f"Stage0{WORKER_REQUEST.replace('.', '_')}"
ROOT5 = Path(os.environ.get("TPMS_FEA_ROOT", default_worker_root)).expanduser().resolve()
INPUT5 = ROOT5/"input"
WORK5 = ROOT5/"work"
LOCAL_EXPORT5 = ROOT5/"production_checkpoints"
for d in [INPUT5, WORK5, LOCAL_EXPORT5]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_NAME = "03_FINAL_population_120_WITH_SPLITS.csv"
REQUIRED_ZIP_HINT = "03_1_FINAL_TPMS_DATASET_DESIGN.zip"

manifest_override = os.environ.get("TPMS_MANIFEST")
manifest_candidates = [Path(manifest_override).expanduser().resolve()] if manifest_override else []
manifest_candidates += list(INPUT5.rglob(MANIFEST_NAME)) + list(Path.cwd().glob(MANIFEST_NAME))

if manifest_candidates:
    manifest_candidates = [p for p in manifest_candidates if p.is_file()]
    if not manifest_candidates:
        raise RuntimeError("TPMS_MANIFEST does not point to an existing file.")
else:
    zip_candidates = list(INPUT5.glob("*.zip")) + list(Path.cwd().glob(REQUIRED_ZIP_HINT))
    if len(zip_candidates) != 1:
        raise RuntimeError(
            f"Set TPMS_MANIFEST or place exactly one {REQUIRED_ZIP_HINT} in "
            f"{INPUT5} or the current directory."
        )
    zp = zip_candidates[0]
    if not zp.exists():
        raise RuntimeError("Uploaded ZIP could not be located.")

    print("Checking ZIP integrity...")
    with zipfile.ZipFile(zp) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at: {bad}")
        names = zf.namelist()
        if not any(Path(n).name == MANIFEST_NAME for n in names):
            raise RuntimeError(
                f"Wrong ZIP: it does not contain {MANIFEST_NAME}"
            )
        zf.extractall(INPUT5)

    manifest_candidates = list(INPUT5.rglob(MANIFEST_NAME))
    if not manifest_candidates:
        raise RuntimeError(
            "ZIP extracted but authoritative manifest was not found."
        )

MANIFEST_FILE = manifest_candidates[0]


SEED = 20260918
np.random.seed(SEED)

ARCHS = ["gyroid","diamond","primitive"]
DOMAIN_MM = np.array([20.0,20.0,20.0])
ORIGIN_MM = np.array([-10.0,-10.0,0.0])
PROD_N = 128
QC_RHO_ABS_ERR_MAX = 0.010

CONNECTIVITY = 26
SHORT_EDGE_THICKNESS_FRAC = 0.35
SPUR_THICKNESS_FRAC = 0.50

E_MPA = 110000.0
NU = 0.33
TARGET_MACRO_STRAIN = 0.01
TOP_DISPLACEMENT_MM = -TARGET_MACRO_STRAIN * DOMAIN_MM[2]
ELEMENT_TYPE = "C3D4"

PRODUCTION_MESH_LABEL = "S250k"
PRODUCTION_MESH_CFG = {
    "minratio": 1.30,
    "mindihedral": 10.0,
    "steinerleft": 250000,
}

PRIMARY_K = 32
IDW_POWER = 2.0

AUTHORIZED_TARGETS = ["von_mises_stress","equivalent_strain"]
FULL_TENSOR_TARGETS_AUTHORIZED = False
PRIMITIVE_CLOSURE_DECISION = "NO_DEFENSIBLE_TENSOR_CONVERGENCE_CLOSURE"

MAX_TETS = 5_000_000
MAX_NODES = 1_300_000
MIN_Q01 = 0.10
MAX_EDGE_RATIO_P99 = 8.0
MIN_RAM_GB_BEFORE_CCX = 30.0
MIN_DISK_GB_BEFORE_CCX = 20.0

MAX_K32_RADIUS_P95_MM = 0.50
MIN_PARSE_FRACTION = 1.0

WORKER_CONFIG = {
    "5.2A": ("primitive", 1), "5.2B": ("primitive", 2),
    "5.2C": ("diamond", 1), "5.2D": ("diamond", 2),
    "5.2E": ("gyroid", 1), "5.2F": ("gyroid", 2),
}
WORKER_NAME = WORKER_REQUEST
if WORKER_NAME not in WORKER_CONFIG:
    raise RuntimeError(f"TPMS_WORKER must be one of {sorted(WORKER_CONFIG)}")
WORKER_ARCHITECTURE, WORKER_HALF = WORKER_CONFIG[WORKER_NAME]
WORKER_EXPECTED_COUNT = 20
DELETE_LARGE_TEMP_AFTER_SUCCESS = True


def hard_gc():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

print(f"FEA worker {WORKER_NAME}: {WORKER_ARCHITECTURE}")

if "MANIFEST_FILE" not in globals() or not Path(MANIFEST_FILE).exists():
    raise RuntimeError("The geometry-design manifest was not staged.")

manifest_file = Path(MANIFEST_FILE)
manifest = pd.read_csv(manifest_file)

required = {
    "sample_id","architecture","target_relative_density","cell_size_mm",
    "grading_mode","grading_amplitude","phase_x_rad","phase_y_rad","phase_z_rad",
    "iid_split"
}
missing = required - set(manifest.columns)
if missing:
    raise RuntimeError("Manifest missing columns: " + ", ".join(sorted(missing)))
if len(manifest)!=120 or not manifest.sample_id.is_unique:
    raise RuntimeError("Authoritative 120-sample manifest identity check failed.")
if set(manifest.architecture)!=set(ARCHS):
    raise RuntimeError("Manifest architecture identity check failed.")
counts=manifest.groupby("architecture").size().reindex(ARCHS)
if not (counts==40).all():
    raise RuntimeError(f"Expected 40 per architecture; got {counts.to_dict()}")

architecture_rows = manifest[manifest.architecture==WORKER_ARCHITECTURE].copy()
if len(architecture_rows)!=40:
    raise RuntimeError(f"Expected exactly 40 {WORKER_ARCHITECTURE} samples.")

start = 0 if WORKER_HALF == 1 else 20
WORKER_IDS = architecture_rows.iloc[start:start+20].sample_id.astype(str).tolist()
worker_manifest = manifest[manifest.sample_id.astype(str).isin(WORKER_IDS)].copy()
worker_manifest["_worker_order"] = pd.Categorical(
    worker_manifest.sample_id.astype(str),
    categories=WORKER_IDS,
    ordered=True
)
worker_manifest = worker_manifest.sort_values("_worker_order").drop(columns="_worker_order").reset_index(drop=True)

if len(WORKER_IDS)!=20 or len(set(WORKER_IDS))!=20:
    raise RuntimeError(f"{WORKER_NAME} assignment must contain exactly 20 unique IDs.")
if len(worker_manifest)!=20:
    raise RuntimeError(f"{WORKER_NAME} manifest selection did not resolve exactly 20 rows.")
if set(worker_manifest.architecture)!={WORKER_ARCHITECTURE}:
    raise RuntimeError(f"{WORKER_NAME} contains the wrong architecture.")
if worker_manifest.sample_id.astype(str).tolist()!=WORKER_IDS:
    raise RuntimeError(f"{WORKER_NAME} assignment order/identity mismatch.")

DRIVE_PERSISTENT = False
EXPORT5 = LOCAL_EXPORT5




def tpms_field(arch,X,Y,Z,cell_mm,phase):
    k=2*np.pi/cell_mm
    x=k*X+phase[0]; y=k*Y+phase[1]; z=k*Z+phase[2]
    if arch=='gyroid':
        return np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.sin(z)*np.cos(x)
    if arch=='diamond':
        return (np.sin(x)*np.sin(y)*np.sin(z)+np.sin(x)*np.cos(y)*np.cos(z)+
                np.cos(x)*np.sin(y)*np.cos(z)+np.cos(x)*np.cos(y)*np.sin(z))
    if arch=='primitive': return np.cos(x)+np.cos(y)+np.cos(z)
    raise ValueError(arch)

def grading_profile(z_norm,mode):
    if mode=='uniform': return np.zeros_like(z_norm)
    if mode=='linear': return 2*z_norm-1
    if mode=='sinusoidal': return np.sin(2*np.pi*z_norm)
    raise ValueError(mode)

def make_grid(n):
    spacing=DOMAIN_MM/n
    xs=ORIGIN_MM[0]+(np.arange(n)+.5)*spacing[0]
    ys=ORIGIN_MM[1]+(np.arange(n)+.5)*spacing[1]
    zs=ORIGIN_MM[2]+(np.arange(n)+.5)*spacing[2]
    return xs,ys,zs,spacing

def calibrate_mask(row,n=PROD_N,tol=2e-4,max_iter=40):
    xs,ys,zs,spacing=make_grid(n)
    X=xs[:,None,None]; Y=ys[None,:,None]; Z=zs[None,None,:]
    phase=np.array([row.phase_x_rad,row.phase_y_rad,row.phase_z_rad])
    F=tpms_field(row.architecture,X,Y,Z,row.cell_size_mm,phase).astype(np.float32)
    absF=np.abs(F)
    z_norm=((zs-ORIGIN_MM[2])/DOMAIN_MM[2])[None,None,:]
    g=grading_profile(z_norm,row.grading_mode).astype(np.float32)
    amp=float(row.grading_amplitude); target=float(row.target_relative_density)
    lo=0.0; hi=float(np.quantile(absF,min(.95,max(.60,target+.30))))+1e-6
    def density_at(base):
        thr=base*np.clip(1+amp*g,.20,None); return float(np.mean(absF<=thr))
    while density_at(hi)<target:
        hi*=1.5
        if hi>float(absF.max())*2+1: raise RuntimeError('Could not bracket density')
    for _ in range(max_iter):
        mid=.5*(lo+hi); rho=density_at(mid)
        if abs(rho-target)<=tol: break
        if rho<target: lo=mid
        else: hi=mid
    base=.5*(lo+hi); thr=base*np.clip(1+amp*g,.20,None)
    mask=absF<=thr
    return mask,float(base),float(mask.mean()),spacing

def surface_from_mask(mask,spacing):
    padded=np.pad(mask.astype(np.uint8),1)
    verts,faces,_,_=marching_cubes(padded,level=.5,spacing=tuple(spacing))
    verts-=spacing[None,:]; verts+=ORIGIN_MM[None,:]
    return trimesh.Trimesh(vertices=verts,faces=faces,process=False)

from collections import defaultdict, deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

def neighbor_offsets(connectivity=26):
    out=[]
    for a in (-1,0,1):
        for b in (-1,0,1):
            for c in (-1,0,1):
                if a==b==c==0: continue
                man=abs(a)+abs(b)+abs(c)
                if connectivity==6 and man==1: out.append((a,b,c))
                elif connectivity==18 and man<=2: out.append((a,b,c))
                elif connectivity==26: out.append((a,b,c))
    return out

def voxelize_filled(mesh, target_resolution):
    ext=np.asarray(mesh.extents,float)
    pitch=float(ext.max()/target_resolution)
    vg=mesh.voxelized(pitch=pitch).fill()
    solid=np.asarray(vg.matrix,dtype=bool)
    T=np.asarray(vg.transform,float)
    return solid,T,pitch

def skeletonize_3d_compat(solid):
    return skeletonize(solid).astype(bool)

def skel_graph(skel, connectivity=26):
    coords=np.argwhere(skel)
    lut={tuple(p):i for i,p in enumerate(coords)}
    adj=[set() for _ in range(len(coords))]
    for i,p in enumerate(coords):
        for d in neighbor_offsets(connectivity):
            q=(int(p[0]+d[0]),int(p[1]+d[1]),int(p[2]+d[2]))
            j=lut.get(q)
            if j is not None and j!=i:
                adj[i].add(j)
    return coords,adj

def ijk_to_world(ijk,T):
    P=np.c_[np.asarray(ijk,float),np.ones(len(ijk))]
    return (P@T.T)[:,:3]

def connected_sets(nodes, adj):
    nodes=set(nodes); seen=set(); comps=[]
    for s in list(nodes):
        if s in seen: continue
        q=[s]; seen.add(s); comp=[]
        while q:
            u=q.pop(); comp.append(u)
            for v in adj[u]:
                if v in nodes and v not in seen:
                    seen.add(v); q.append(v)
        comps.append(comp)
    return comps

def contract_junction_regions(coords, adj, world):
    deg=np.array([len(a) for a in adj],int)
    jvox=np.where(deg>2)[0]
    jcomps=connected_sets(jvox,adj)

    supernodes=[]
    voxel_to_super={}
    for comp in jcomps:
        sid=len(supernodes)
        supernodes.append({"kind":"junction","voxels":list(comp),
                           "pos":world[comp].mean(axis=0)})
        for v in comp: voxel_to_super[v]=sid

    for v in np.where(deg==1)[0]:
        sid=len(supernodes)
        supernodes.append({"kind":"endpoint","voxels":[int(v)],"pos":world[v].copy()})
        voxel_to_super[int(v)]=sid

    for v in np.where(deg==0)[0]:
        sid=len(supernodes)
        supernodes.append({"kind":"isolated","voxels":[int(v)],"pos":world[v].copy()})
        voxel_to_super[int(v)]=sid

    return supernodes,voxel_to_super,deg

def trace_region_branches(coords, adj, world, supernodes, voxel_to_super, deg):
    visited=set(); branches=[]
    for sid,sn in enumerate(supernodes):
        for start_vox in sn["voxels"]:
            for nxt in adj[start_vox]:
                if voxel_to_super.get(nxt)==sid:  # internal junction-region connection
                    continue
                key=tuple(sorted((start_vox,nxt)))
                if key in visited: continue
                visited.add(key)
                path=[start_vox,nxt]
                prev=start_vox; cur=nxt
                end_sid=voxel_to_super.get(cur)

                while end_sid is None:
                    candidates=[x for x in adj[cur] if x!=prev]
                    if not candidates: break
                    if len(candidates)>1:
                        break
                    nn=candidates[0]
                    visited.add(tuple(sorted((cur,nn))))
                    path.append(nn); prev,cur=cur,nn
                    end_sid=voxel_to_super.get(cur)

                if end_sid is None:
                    end_sid=len(supernodes)
                    supernodes.append({"kind":"synthetic_terminal","voxels":[int(cur)],
                                       "pos":world[cur].copy()})
                    voxel_to_super[int(cur)]=end_sid

                if end_sid==sid:
                    continue

                pts=world[path]
                seg=np.linalg.norm(np.diff(pts,axis=0),axis=1)
                plen=float(seg.sum())
                chord=float(np.linalg.norm(supernodes[end_sid]["pos"]-supernodes[sid]["pos"]))
                branches.append({"source":sid,"target":end_sid,"path_voxels":path,
                                 "path_length":plen,"chord_length":chord})
    return supernodes,branches

def branch_dataframe(supernodes, branches, edt, coords, pitch):
    rows=[]
    for b in branches:
        s,t=b["source"],b["target"]
        p0=np.asarray(supernodes[s]["pos"]); p1=np.asarray(supernodes[t]["pos"])
        dv=p1-p0; chord=float(np.linalg.norm(dv))
        theta=float(np.arctan2(dv[1],dv[0]))
        phi=float(np.arccos(np.clip(dv[2]/chord,-1,1))) if chord>0 else 0.0
        vals=[]
        for vi in b["path_voxels"]:
            ijk=coords[vi]
            vals.append(2.0*float(edt[tuple(ijk)])*pitch)
        rows.append({
            "source":s,"target":t,
            "path_length":float(b["path_length"]),
            "chord_length":chord,
            "tortuosity":float(b["path_length"]/chord) if chord>0 else np.nan,
            "theta":theta,"phi":phi,
            "local_thickness_mean":float(np.mean(vals)) if vals else np.nan,
            "local_thickness_min":float(np.min(vals)) if vals else np.nan,
            "voxel_count":len(b["path_voxels"])
        })
    return pd.DataFrame(rows)

def node_dataframe(supernodes):
    return pd.DataFrame([{
        "node":i,"x":float(n["pos"][0]),"y":float(n["pos"][1]),"z":float(n["pos"][2]),
        "kind":n["kind"],"region_voxels":len(n["voxels"])
    } for i,n in enumerate(supernodes)])

def dedupe_edges(nodes, edges):
    if len(edges)==0: return nodes.copy(),edges.copy()
    e=edges.copy()
    e=e[e.source!=e.target].copy()
    e["a"]=e[["source","target"]].min(axis=1)
    e["b"]=e[["source","target"]].max(axis=1)
    e=e.sort_values(["a","b","path_length"]).drop_duplicates(["a","b"],keep="first")
    e=e.drop(columns=["a","b"]).reset_index(drop=True)
    return nodes.copy(),e

def reindex_graph(nodes, edges):
    used=sorted(set(edges.source.astype(int)).union(set(edges.target.astype(int)))) if len(edges) else []
    mp={old:i for i,old in enumerate(used)}
    n=nodes[nodes.node.isin(used)].copy()
    n["node"]=n["node"].map(mp)
    n=n.sort_values("node").reset_index(drop=True)
    e=edges.copy()
    if len(e):
        e["source"]=e["source"].map(mp).astype(int); e["target"]=e["target"].map(mp).astype(int)
    return n,e.reset_index(drop=True)

def graph_degrees(nodes,edges):
    d=np.zeros(len(nodes),int)
    for s,t in edges[["source","target"]].to_numpy(int):
        d[s]+=1; d[t]+=1
    return d

def contract_short_edges(nodes, edges, threshold):
    if len(edges)==0: return nodes,edges
    parent=list(range(len(nodes)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a

    short=edges[edges.path_length < threshold]
    for s,t in short[["source","target"]].to_numpy(int): union(s,t)

    groups=defaultdict(list)
    for i in range(len(nodes)): groups[find(i)].append(i)
    old2new={old:k for k,g in enumerate(groups.values()) for old in g}
    newrows=[]
    for k,g in enumerate(groups.values()):
        sub=nodes.iloc[g]
        weights=np.maximum(sub["region_voxels"].to_numpy(float),1)
        xyz=np.average(sub[["x","y","z"]].to_numpy(float),axis=0,weights=weights)
        kinds=set(sub["kind"])
        kind="junction" if "junction" in kinds or len(g)>1 else sub.iloc[0]["kind"]
        newrows.append({"node":k,"x":xyz[0],"y":xyz[1],"z":xyz[2],
                        "kind":kind,"region_voxels":int(sub.region_voxels.sum())})
    nn=pd.DataFrame(newrows)

    ee=edges.copy()
    ee["source"]=ee["source"].map(old2new); ee["target"]=ee["target"].map(old2new)
    nn,ee=dedupe_edges(nn,ee)
    nn,ee=reindex_graph(nn,ee)
    return nn,ee

def prune_terminal_spurs(nodes,edges,threshold,max_iter=10):
    n,e=nodes.copy(),edges.copy()
    for _ in range(max_iter):
        if len(e)==0: break
        d=graph_degrees(n,e)
        kill=[]
        for idx,r in e.iterrows():
            s,t=int(r.source),int(r.target)
            if r.path_length < threshold and (d[s]==1 or d[t]==1):
                kill.append(idx)
        if not kill: break
        e=e.drop(index=kill).reset_index(drop=True)
        n,e=reindex_graph(n,e)
    return n,e

def recompute_edge_geometry(nodes,edges):
    e=edges.copy()
    if len(e)==0: return e
    P=nodes[["x","y","z"]].to_numpy(float)
    vals=[]
    for _,r in e.iterrows():
        s,t=int(r.source),int(r.target)
        dv=P[t]-P[s]; chord=float(np.linalg.norm(dv))
        theta=float(np.arctan2(dv[1],dv[0]))
        phi=float(np.arccos(np.clip(dv[2]/chord,-1,1))) if chord>0 else 0.0
        plen=max(float(r.path_length),chord)
        vals.append((chord,plen,plen/chord if chord>0 else np.nan,theta,phi))
    e[["chord_length","path_length","tortuosity","theta","phi"]]=np.asarray(vals,float)
    return e

def graph_components(nodes,edges):
    if len(nodes)==0: return 0
    if len(edges)==0: return len(nodes)
    ij=np.vstack([edges[["source","target"]].to_numpy(int),
                  edges[["target","source"]].to_numpy(int)])
    A=csr_matrix((np.ones(len(ij)),(ij[:,0],ij[:,1])),shape=(len(nodes),len(nodes)))
    return int(connected_components(A,directed=False,return_labels=False))

def spectral_signature(nodes,edges,k=12):
    if len(nodes)<3 or len(edges)==0: return np.array([])
    ij=np.vstack([edges[["source","target"]].to_numpy(int),
                  edges[["target","source"]].to_numpy(int)])
    A=csr_matrix((np.ones(len(ij)),(ij[:,0],ij[:,1])),shape=(len(nodes),len(nodes)))
    d=np.asarray(A.sum(axis=1)).ravel()
    inv=np.zeros_like(d,float); inv[d>0]=1/np.sqrt(d[d>0])
    D=csr_matrix((inv,(np.arange(len(d)),np.arange(len(d)))),shape=A.shape)
    L=csr_matrix(np.eye(len(d)))-D@A@D
    kk=min(k+1,len(nodes)-1)
    try:
        vals=np.sort(eigsh(L,k=kk,which="SM",return_eigenvectors=False))
        return vals[1:min(len(vals),k+1)]
    except Exception:
        return np.array([])



def analyze_mask_graph(solid,spacing):
    pitch=float(spacing[0])
    T=np.eye(4); T[0,0]=spacing[0]; T[1,1]=spacing[1]; T[2,2]=spacing[2]
    T[:3,3]=ORIGIN_MM+0.5*spacing
    skel=skeletonize_3d_compat(solid)
    coords,adj=skel_graph(skel,CONNECTIVITY)
    world=ijk_to_world(coords,T)
    edt=ndimage.distance_transform_edt(solid)
    supernodes,v2s,deg=contract_junction_regions(coords,adj,world)
    supernodes,branches=trace_region_branches(coords,adj,world,supernodes,v2s,deg)
    nodes=node_dataframe(supernodes)
    edges=branch_dataframe(supernodes,branches,edt,coords,pitch)
    nodes,edges=dedupe_edges(nodes,edges); nodes,edges=reindex_graph(nodes,edges)
    med=float(np.nanmedian(edges.local_thickness_mean)) if len(edges) else np.nan
    nodes,edges=contract_short_edges(nodes,edges,SHORT_EDGE_THICKNESS_FRAC*med)
    nodes,edges=prune_terminal_spurs(nodes,edges,SPUR_THICKNESS_FRAC*med)
    nodes,edges=dedupe_edges(nodes,edges); nodes,edges=reindex_graph(nodes,edges)
    edges=recompute_edge_geometry(nodes,edges)
    nodes['degree']=graph_degrees(nodes,edges)
    return nodes,edges,{
        'graph_nodes':len(nodes),'graph_edges':len(edges),'graph_components':graph_components(nodes,edges),
        'endpoints':int((nodes.degree==1).sum()),'junctions':int((nodes.degree>=3).sum()),
        'median_local_thickness_mm':med,'skeleton_voxels':int(skel.sum())}


def tet_volumes(p,t):
    P=p[t]
    return np.abs(np.einsum(
        "ij,ij->i",
        P[:,1]-P[:,0],
        np.cross(P[:,2]-P[:,0],P[:,3]-P[:,0])
    ))/6.0

def mesh_qc(p,t):
    P=p[t]
    vol=tet_volumes(p,t)
    ee=np.stack([
        np.linalg.norm(P[:,0]-P[:,1],axis=1),
        np.linalg.norm(P[:,0]-P[:,2],axis=1),
        np.linalg.norm(P[:,0]-P[:,3],axis=1),
        np.linalg.norm(P[:,1]-P[:,2],axis=1),
        np.linalg.norm(P[:,1]-P[:,3],axis=1),
        np.linalg.norm(P[:,2]-P[:,3],axis=1)
    ],axis=1)
    q=12*(3*vol)**(2/3)/np.maximum((ee**2).sum(1),1e-30)
    er=ee.max(1)/np.maximum(ee.min(1),1e-30)
    return vol,q,er

def exterior_faces(t):
    ff=np.vstack([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[1,2,3]],t[:,[2,0,3]]])
    ss=np.sort(ff,axis=1)
    order=np.lexsort((ss[:,2],ss[:,1],ss[:,0]))
    s=ss[order]
    same=np.all(s[1:]==s[:-1],axis=1)
    starts=np.r_[0,np.where(~same)[0]+1]
    ends=np.r_[starts[1:],len(s)]
    return ff[order[starts[(ends-starts)==1]]]

def end_sets(p,t,pitch):
    bf=exterior_faces(t)
    tri=p[bf]
    c=tri.mean(1)
    nv=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    nz=np.abs((nv/np.maximum(np.linalg.norm(nv,axis=1)[:,None],1e-15))[:,2])
    z0,z1=p[:,2].min(),p[:,2].max()
    tol=1.5*pitch
    bot=np.unique(bf[(c[:,2]<=z0+tol)&(nz>=.80)])
    top=np.unique(bf[(c[:,2]>=z1-tol)&(nz>=.80)])
    return bf,bot,top



FLOAT_PAT = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)

def nums(line):
    return [
        float(x.replace("D", "E").replace("d", "e"))
        for x in FLOAT_PAT.findall(line)
    ]

def write_ids_zero_based(f, ids, n=16):
    """Write zero-based Python IDs as one-based CalculiX IDs."""
    z = np.asarray(ids, dtype=np.int64) + 1
    for i in range(0, len(z), n):
        f.write(",".join(map(str, z[i:i+n])) + "\n")

def rigid_body_anchors(p, bottom):
    bottom = np.asarray(bottom, np.int64)
    xy = p[bottom, :2]
    center = xy.mean(axis=0)
    a = int(bottom[np.argmin(np.linalg.norm(xy-center, axis=1))])
    b = int(
        bottom[
            np.argmax(
                np.linalg.norm(p[bottom, :2] - p[a, :2], axis=1)
            )
        ]
    )
    if a == b:
        raise RuntimeError("Rigid-body anchors collapsed.")
    return a, b

def von_mises(A):
    a,b,c,d,e,f = [A[:,i].astype(np.float64) for i in range(6)]
    return np.sqrt(
        .5*((a-b)**2 + (b-c)**2 + (c-a)**2)
        + 3*(d*d + e*e + f*f)
    )

def equivalent_strain(E):
    ex,ey,ez,gxy,gxz,gyz = [
        E[:,i].astype(np.float64) for i in range(6)
    ]
    m = (ex+ey+ez)/3.0
    return np.sqrt(
        (2/3) * (
            (ex-m)**2 + (ey-m)**2 + (ez-m)**2
            + 2*((gxy/2)**2 + (gxz/2)**2 + (gyz/2)**2)
        )
    )

def parse_target_dat(dat_path, top1, target_elem1):
    """
    Parse:
      - top reaction vector
      - six stress components for TARGETS
      - six strain components for TARGETS

    Output arrays preserve target_elem1 order.
    """
    dat_path = Path(dat_path)
    top_set = set(map(int, np.asarray(top1, np.int64)))
    elem_order = np.asarray(target_elem1, np.int64)
    elem_set = set(map(int, elem_order))

    rf = {}
    stress = {}
    strain = {}
    mode = None

    with open(dat_path, "r", errors="ignore") as f:
        for line in f:
            lo = line.lower()

            if "force" in lo and ("fx" in lo or "rf" in lo or "node" in lo):
                mode = "rf"
                continue

            if "stress" in lo and ("sxx" in lo or "elem" in lo):
                mode = "stress"
                continue

            if "strain" in lo and ("exx" in lo or "elem" in lo):
                mode = "strain"
                continue

            vals = nums(line)

            if mode == "rf" and len(vals) >= 4:
                nid = int(round(vals[0]))
                if nid in top_set:
                    rf[nid] = np.asarray(vals[-3:], float)

            elif mode in ("stress", "strain") and len(vals) >= 8:
                eid = int(round(vals[0]))
                if eid in elem_set:
                    arr = np.asarray(vals[-6:], float)
                    if mode == "stress":
                        stress[eid] = arr
                    else:
                        strain[eid] = arr

    if not rf:
        raise RuntimeError("No top reaction rows parsed.")

    total_rf = np.sum(np.asarray(list(rf.values())), axis=0)

    S = np.full((len(elem_order), 6), np.nan, float)
    E = np.full((len(elem_order), 6), np.nan, float)

    for i, eid in enumerate(elem_order):
        if int(eid) in stress:
            S[i] = stress[int(eid)]
        if int(eid) in strain:
            E[i] = strain[int(eid)]

    return total_rf, S, E

def read_elset(deck_path, name="TARGETS"):
    ids = []
    active = False
    target = f"*ELSET,ELSET={name}".replace(" ", "").upper()

    with open(deck_path, "r") as f:
        for line in f:
            s = line.strip()
            norm = s.replace(" ", "").upper()

            if norm.startswith("*"):
                if active:
                    break
                active = (norm == target)
                continue

            if active and s:
                for token in s.split(","):
                    token = token.strip()
                    if token:
                        ids.append(int(token))

    return np.asarray(ids, np.int64)



def read_id_set(deck_path, keyword, name):
    """Read an explicit NSET or ELSET block from a CalculiX deck."""
    ids=[]; active=False
    target=f"*{keyword},{keyword}={name}".replace(" ","").upper()
    with open(deck_path,"r") as f:
        for line in f:
            s=line.strip(); norm=s.replace(" ","").upper()
            if norm.startswith("*"):
                if active: break
                active=(norm==target); continue
            if active and s:
                ids.extend(int(tok.strip()) for tok in s.split(",") if tok.strip())
    return np.asarray(ids,np.int64)

def sha256_array(a):
    import hashlib
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()

def validate_checkpoint(run_dir, intended_elem1):
    run_dir=Path(run_dir)
    sp=run_dir/"target_solve_summary.json"
    sf=run_dir/"target_stress.npy"
    ef=run_dir/"target_strain.npy"
    ip=run_dir/"target_elem1.npy"
    if not all(x.exists() for x in [sp,sf,ef,ip]): return None
    try:
        meta=json.loads(sp.read_text())
        ids=np.load(ip); S=np.load(sf,mmap_mode="r"); E=np.load(ef,mmap_mode="r")
        ok=(meta.get("solver_completed") is True and
            np.array_equal(ids,np.asarray(intended_elem1,np.int64)) and
            S.shape==(len(ids),6) and E.shape==(len(ids),6) and
            np.isfinite(S).all() and np.isfinite(E).all())
        return meta if ok else None
    except Exception:
        return None


def idw_from_unique(unique_elem0, values_unique, idx, dist, power=2.0):
    unique_elem0=np.asarray(unique_elem0,np.int64)
    idx=np.asarray(idx,np.int64)
    dist=np.asarray(dist,np.float64)
    loc=np.searchsorted(unique_elem0,idx)
    if np.any(loc>=len(unique_elem0)) or not np.array_equal(unique_elem0[loc],idx):
        raise RuntimeError("Exact target-element lookup failed.")
    vals=np.asarray(values_unique)[loc]
    w=1.0/np.maximum(dist,1e-9)**power
    w/=w.sum(axis=1,keepdims=True)
    if vals.ndim==3:
        return np.sum(w[:,:,None]*vals,axis=1)
    return np.sum(w*vals,axis=1)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def checkpoint_dir(sid):
    return EXPORT5/sid

def checkpoint_complete(sid):
    d=checkpoint_dir(sid)
    req=[
        d/"metadata.json",
        d/"canonical_nodes.csv",
        d/"canonical_edges.csv",
        d/"graph_targets_scalar.npz",
        d/"mapping_k32.npz",
        d/"target_scalar_elements.npz",
    ]
    if not all(p.exists() for p in req):
        return False
    try:
        meta=json.loads((d/"metadata.json").read_text())
        z=np.load(d/"graph_targets_scalar.npz")
        ok=(
            meta.get("checkpoint_verified") is True and
            meta.get("solver_completed") is True and
            meta.get("mesh_label")=="S250k" and
            meta.get("mapping_k")==32 and
            meta.get("authorized_targets")==AUTHORIZED_TARGETS and
            z["y_vm"].ndim==1 and z["y_eq_strain"].ndim==1 and
            len(z["y_vm"])==meta["graph_nodes"] and
            len(z["y_eq_strain"])==meta["graph_nodes"] and
            np.isfinite(z["y_vm"]).all() and
            np.isfinite(z["y_eq_strain"]).all()
        )
        return bool(ok)
    except Exception:
        return False



SMOKE=ROOT5/"smoke_test"
if SMOKE.exists(): shutil.rmtree(SMOKE)
SMOKE.mkdir(parents=True)

p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
t=np.array([[0,1,3,4],[1,2,3,6],[1,3,4,6],[1,4,5,6],[3,4,6,7]],np.int64)
bottom=np.array([0,1,2,3],np.int64)
top=np.array([4,5,6,7],np.int64)
target0=np.arange(len(t),dtype=np.int64)
target1=target0+1
deck=SMOKE/"model.inp"

with open(deck,"w") as f:
    f.write("*HEADING\nStage 05 parser smoke test\n*NODE\n")
    for i,xyz in enumerate(p,1):
        f.write(f"{i},{xyz[0]},{xyz[1]},{xyz[2]}\n")
    f.write("*ELEMENT,TYPE=C3D4,ELSET=SOLID\n")
    for i,q0 in enumerate(t,1):
        q=q0+1
        f.write(f"{i},{q[0]},{q[1]},{q[2]},{q[3]}\n")
    for name,arr in [("BOTTOM",bottom),("TOP",top),("ANCHOR_A",[0]),("ANCHOR_B",[1])]:
        f.write(f"*NSET,NSET={name}\n")
        write_ids_zero_based(f,arr)
    f.write("*ELSET,ELSET=TARGETS\n")
    write_ids_zero_based(f,target0)
    f.write("*MATERIAL,NAME=TI64\n*ELASTIC\n110000.,0.33\n")
    f.write("*SOLID SECTION,ELSET=SOLID,MATERIAL=TI64\n")
    f.write("*STEP\n*STATIC\n*BOUNDARY\n")
    f.write("BOTTOM,3,3,0.\nTOP,3,3,-0.01\nANCHOR_A,1,2,0.\nANCHOR_B,2,2,0.\n")
    f.write("*NODE PRINT,NSET=TOP\nRF\n")
    f.write("*EL PRINT,ELSET=TARGETS\nS\nE\n*END STEP\n")

assert np.array_equal(read_id_set(deck,"ELSET","TARGETS"),target1)
assert np.array_equal(read_id_set(deck,"NSET","TOP"),top+1)

with open(SMOKE/"ccx_stdout.txt","w") as out, open(SMOKE/"ccx_stderr.txt","w") as err:
    cp=subprocess.run([CCX_PATH,"model"],cwd=SMOKE,stdout=out,stderr=err,text=True)
if cp.returncode!=0:
    raise RuntimeError("Tiny CalculiX smoke solve failed — STOP.")
rf,S,E=parse_target_dat(SMOKE/"model.dat",top+1,target1)
if S.shape!=(5,6) or E.shape!=(5,6) or not np.isfinite(S).all() or not np.isfinite(E).all():
    raise RuntimeError("S/E parser smoke test failed — STOP.")
if not np.isfinite(rf).all():
    raise RuntimeError("Reaction parser smoke test failed — STOP.")

PARSER_SMOKE_PASS=True
print("CalculiX parser smoke test passed")

if not globals().get("PARSER_SMOKE_PASS",False):
    raise RuntimeError("Solver/parser smoke test must pass first.")

def process_one_sample(row):
    sid=str(row.sample_id)
    arch=str(row.architecture)
    final_dir=checkpoint_dir(sid)
    if checkpoint_complete(sid):
        print(f"✓ {sid} already has a verified checkpoint — SKIP")
        return {"sample_id":sid,"architecture":arch,"status":"SKIPPED_VERIFIED"}

    run_dir=WORK5/sid
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True,exist_ok=True)

    t_start=time.time()
    stage="start"
    try:
        print(f"PRODUCTION — {sid} — {arch.upper()}")

        stage="geometry"
        solid,tau,rho,spacing=calibrate_mask(row,PROD_N)
        rho_err=abs(rho-float(row.target_relative_density))
        comps=int(ndimage.label(solid,structure=np.ones((3,3,3),np.uint8))[1])
        if rho_err>QC_RHO_ABS_ERR_MAX or comps!=1:
            raise RuntimeError(f"Geometry QC failed: rho_err={rho_err}, components={comps}")

        surf=surface_from_mask(solid,spacing)
        if not surf.is_watertight or not surf.is_winding_consistent:
            raise RuntimeError("Surface watertight/winding QC failed.")

        stage="graph"
        nodes,edges,gmeta=analyze_mask_graph(solid,spacing)
        if gmeta["graph_components"]!=1 or len(nodes)==0 or len(edges)==0:
            raise RuntimeError("Canonical graph QC failed.")
        gpos=nodes[["x","y","z"]].to_numpy(np.float64)

        nodes.to_csv(final_dir/"canonical_nodes.csv",index=False)
        edges.to_csv(final_dir/"canonical_edges.csv",index=False)

        stage="tetgen"
        tg=tetgen.TetGen(
            np.asarray(surf.vertices,np.float64),
            np.asarray(surf.faces,np.int32)
        )
        tm=time.time()
        res=tg.tetrahedralize(order=1,quality=True,**PRODUCTION_MESH_CFG)
        mesh_s=time.time()-tm
        p=np.asarray(res[0],np.float64)
        t=np.asarray(res[1],np.int64)
        n_nodes,n_tets=len(p),len(t)
        print(f"S250k mesh: {n_nodes:,} nodes | {n_tets:,} tets | {mesh_s/60:.2f} min")

        if n_nodes>MAX_NODES or n_tets>MAX_TETS:
            raise RuntimeError("Hard mesh resource gate failed.")

        stage="mesh_qc"
        vol,q,er=mesh_qc(p,t)
        zero=int((vol<=1e-14).sum())
        q01=float(np.quantile(q,.01))
        qmed=float(np.median(q))
        er99=float(np.quantile(er,.99))
        del vol,q,er
        hard_gc()
        if zero!=0 or q01<MIN_Q01 or er99>MAX_EDGE_RATIO_P99:
            raise RuntimeError(
                f"Mesh quality failed: zero={zero}, Q01={q01:.4f}, ER99={er99:.4f}"
            )

        pitch=float(DOMAIN_MM[0]/PROD_N)
        bf,bottom,top=end_sets(p,t,pitch)
        exterior_count=len(bf)
        del bf
        if len(bottom)<=100 or len(top)<=100:
            raise RuntimeError("Boundary set QC failed.")

        stage="mapping_plan"
        print(f"Computing centroids for {n_tets:,} tetrahedra...")
        cent=(p[t[:,0]]+p[t[:,1]]+p[t[:,2]]+p[t[:,3]])/4.0
        tree=cKDTree(cent)
        dist32,idx32=tree.query(gpos,k=PRIMARY_K,workers=-1)
        if dist32.ndim==1:
            dist32=dist32[:,None]
            idx32=idx32[:,None]
        idx32=np.asarray(idx32,np.int64)
        dist32=np.asarray(dist32,np.float64)
        unique0=np.unique(idx32.ravel())
        k32_p95=float(np.quantile(dist32[:,-1],.95))
        nearest_p95=float(np.quantile(dist32[:,0],.95))
        print(
            f"graph nodes={len(gpos):,} | unique TARGETS={len(unique0):,} "
            f"({100*len(unique0)/n_tets:.3f}% mesh) | k32 radius p95={k32_p95:.4f} mm"
        )
        if k32_p95>MAX_K32_RADIUS_P95_MM:
            raise RuntimeError("k32 mapping-radius gate failed.")

        del cent,tree
        hard_gc()

        stage="deck"
        target1=unique0+1
        anchor_a,anchor_b=rigid_body_anchors(p,bottom)
        deck=run_dir/"model.inp"
        td=time.time()
        with open(deck,"w",buffering=1024*1024) as f:
            f.write(f"*HEADING\nStage {WORKER_NAME} production {sid}\n*NODE\n")
            for i,xyz in enumerate(p,1):
                f.write(f"{i},{xyz[0]:.8g},{xyz[1]:.8g},{xyz[2]:.8g}\n")
            f.write("*ELEMENT,TYPE=C3D4,ELSET=SOLID\n")
            for i,tet in enumerate(t,1):
                qq=tet+1
                f.write(f"{i},{qq[0]},{qq[1]},{qq[2]},{qq[3]}\n")
            for name,arr in [
                ("BOTTOM",bottom),("TOP",top),
                ("ANCHOR_A",[anchor_a]),("ANCHOR_B",[anchor_b])
            ]:
                f.write(f"*NSET,NSET={name}\n")
                write_ids_zero_based(f,arr)
            f.write("*ELSET,ELSET=TARGETS\n")
            write_ids_zero_based(f,unique0)
            f.write(f"*MATERIAL,NAME=TI64\n*ELASTIC\n{E_MPA},{NU}\n")
            f.write("*SOLID SECTION,ELSET=SOLID,MATERIAL=TI64\n")
            f.write("*STEP\n*STATIC\n*BOUNDARY\n")
            f.write(f"BOTTOM,3,3,0.\nTOP,3,3,{TOP_DISPLACEMENT_MM}\n")
            f.write("ANCHOR_A,1,2,0.\nANCHOR_B,2,2,0.\n")
            f.write("*NODE PRINT,NSET=TOP\nRF\n")
            f.write("*EL PRINT,ELSET=TARGETS\nS\nE\n*END STEP\n")
        deck_s=time.time()-td

        if not np.array_equal(read_id_set(deck,"ELSET","TARGETS"),target1):
            raise RuntimeError("TARGETS exact-ID audit failed.")
        if not np.array_equal(read_id_set(deck,"NSET","TOP"),top+1):
            raise RuntimeError("TOP exact-ID audit failed.")
        if not np.array_equal(read_id_set(deck,"NSET","BOTTOM"),bottom+1):
            raise RuntimeError("BOTTOM exact-ID audit failed.")

        stage="resource_preflight"
        ram=psutil.virtual_memory().available/1e9
        disk=shutil.disk_usage(run_dir).free/1e9
        print("Available RAM before CCX:",round(ram,2),"GB")
        print("Free disk before CCX:",round(disk,2),"GB")
        if ram<MIN_RAM_GB_BEFORE_CCX:
            raise RuntimeError(f"RAM gate failed: {ram:.2f} GB")
        if disk<MIN_DISK_GB_BEFORE_CCX:
            raise RuntimeError(f"Disk gate failed: {disk:.2f} GB")

        top1=top+1
        del tg,res,surf,solid,p,t
        hard_gc()

        stage="ccx"
        ts=time.time()
        with open(run_dir/"ccx_stdout.txt","w") as out, open(run_dir/"ccx_stderr.txt","w") as err:
            cp=subprocess.run([CCX_PATH,"model"],cwd=run_dir,stdout=out,stderr=err,text=True)
        solve_s=time.time()-ts
        print("ccx return:",cp.returncode,"| solve minutes:",round(solve_s/60,2))
        if cp.returncode!=0:
            raise RuntimeError(f"CalculiX failed with return code {cp.returncode}")

        dat=run_dir/"model.dat"
        if not dat.exists() or dat.stat().st_size==0:
            raise RuntimeError("CalculiX produced no DAT.")

        stage="parse"
        total_rf,S,E=parse_target_dat(dat,top1,target1)
        validS=np.all(np.isfinite(S),axis=1)
        validE=np.all(np.isfinite(E),axis=1)
        print(f"stress parsed: {validS.sum():,}/{len(validS):,}")
        print(f"strain parsed: {validE.sum():,}/{len(validE):,}")
        if validS.mean()<MIN_PARSE_FRACTION or validE.mean()<MIN_PARSE_FRACTION:
            raise RuntimeError("Incomplete target parse.")

        stage="scalar_targets"
        vm_unique=von_mises(S)
        eq_unique=equivalent_strain(E)
        y_vm=idw_from_unique(unique0,vm_unique,idx32,dist32,IDW_POWER)
        y_eq=idw_from_unique(unique0,eq_unique,idx32,dist32,IDW_POWER)

        if not np.isfinite(y_vm).all() or not np.isfinite(y_eq).all():
            raise RuntimeError("Nonfinite graph scalar target.")
        if np.any(y_vm<0) or np.any(y_eq<0):
            raise RuntimeError("Negative invariant scalar target.")

        np.savez_compressed(
            final_dir/"graph_targets_scalar.npz",
            y_vm=y_vm.astype(np.float32),
            y_eq_strain=y_eq.astype(np.float32),
            mapping_k=np.int64(PRIMARY_K),
            idw_power=np.float64(IDW_POWER),
        )
        np.savez_compressed(
            final_dir/"mapping_k32.npz",
            graph_pos=gpos.astype(np.float32),
            idx_elem0=idx32.astype(np.int32),
            dist_mm=dist32.astype(np.float32),
            unique_elem0=unique0.astype(np.int32),
        )
        np.savez_compressed(
            final_dir/"target_scalar_elements.npz",
            unique_elem0=unique0.astype(np.int32),
            von_mises=vm_unique.astype(np.float32),
            equivalent_strain=eq_unique.astype(np.float32),
        )

        meta={
            "sample_id":sid,
            "architecture":arch,
            "iid_split":str(row.iid_split),
            "target_relative_density":float(row.target_relative_density),
            "rho_128":float(rho),
            "rho_abs_error_128":float(rho_err),
            "tau_128":float(tau),
            "surface_faces":int(len(surface_from_mask(calibrate_mask(row,PROD_N)[0],spacing).faces)) if False else None,
            "graph_nodes":int(len(nodes)),
            "graph_edges":int(len(edges)),
            "graph_components":int(gmeta["graph_components"]),
            "mesh_label":"S250k",
            "mesh_cfg":PRODUCTION_MESH_CFG,
            "mesh_nodes":int(n_nodes),
            "mesh_tets":int(n_tets),
            "mesh_q01":float(q01),
            "mesh_q_median":float(qmed),
            "mesh_edge_ratio_p99":float(er99),
            "zero_volume_tets":int(zero),
            "exterior_faces":int(exterior_count),
            "bottom_nodes":int(len(bottom)),
            "top_nodes":int(len(top)),
            "mapping_k":int(PRIMARY_K),
            "idw_power":float(IDW_POWER),
            "target_unique_elements":int(len(unique0)),
            "target_unique_fraction_pct":float(100*len(unique0)/n_tets),
            "nearest_centroid_p95_mm":nearest_p95,
            "k32_radius_p95_mm":k32_p95,
            "authorized_targets":AUTHORIZED_TARGETS,
            "full_tensor_targets_authorized":False,
            "primitive_closure_decision":PRIMITIVE_CLOSURE_DECISION,
            "solver_completed":True,
            "ccx_returncode":int(cp.returncode),
            "solve_seconds":float(solve_s),
            "solve_minutes":float(solve_s/60),
            "reaction_x_N":float(total_rf[0]),
            "reaction_y_N":float(total_rf[1]),
            "reaction_z_N":float(total_rf[2]),
            "apparent_stiffness_N_per_mm":float(abs(total_rf[2]/TOP_DISPLACEMENT_MM)),
            "vm_graph_min_MPa":float(np.min(y_vm)),
            "vm_graph_median_MPa":float(np.median(y_vm)),
            "vm_graph_p95_MPa":float(np.quantile(y_vm,.95)),
            "vm_graph_max_MPa":float(np.max(y_vm)),
            "eq_graph_min":float(np.min(y_eq)),
            "eq_graph_median":float(np.median(y_eq)),
            "eq_graph_p95":float(np.quantile(y_eq,.95)),
            "eq_graph_max":float(np.max(y_eq)),
            "checkpoint_verified":True,
            "runtime_minutes_total":float((time.time()-t_start)/60),
        }
        (final_dir/"metadata.json").write_text(json.dumps(meta,indent=2))

        if not checkpoint_complete(sid):
            raise RuntimeError("Post-save compact checkpoint validation failed.")

        (final_dir/"SUCCESS.txt").write_text(
            f"{sid} verified production checkpoint for worker {WORKER_NAME}\n"
        )

        print("✓ VERIFIED COMPACT PRODUCTION CHECKPOINT:", final_dir)

        if DELETE_LARGE_TEMP_AFTER_SUCCESS:
            shutil.rmtree(run_dir,ignore_errors=True)

        hard_gc()
        return {
            "sample_id":sid,"architecture":arch,"status":"COMPLETE",
            "mesh_tets":n_tets,"graph_nodes":len(nodes),
            "solve_minutes":solve_s/60,
            "vm_p95":float(np.quantile(y_vm,.95)),
            "eq_p95":float(np.quantile(y_eq,.95)),
        }

    except Exception as e:
        fail={
            "sample_id":sid,
            "architecture":arch,
            "stage":stage,
            "error_type":type(e).__name__,
            "error":str(e),
            "time":time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (final_dir/"FAILURE.json").write_text(json.dumps(fail,indent=2))
        print("✗ SAMPLE FAILED AT STAGE:",stage)
        print(type(e).__name__+":",e)
        print("Failure record:",final_dir/"FAILURE.json")
        hard_gc()
        return {"sample_id":sid,"architecture":arch,"status":"FAILED","stage":stage,"error":str(e)}


if worker_manifest.sample_id.astype(str).tolist() != WORKER_IDS:
    raise RuntimeError("Worker assignment changed in memory — STOP.")

completed = {sid for sid in WORKER_IDS if checkpoint_complete(sid)}
remaining_ids = [sid for sid in WORKER_IDS if sid not in completed]

print(f"Verified complete before this run: {len(completed)}/20")
print(f"Remaining assigned samples: {len(remaining_ids)}")

session_results=[]
for _,row in worker_manifest.iterrows():
    sid=str(row.sample_id)
    if sid in completed:
        session_results.append({
            "sample_id":sid,
            "architecture":WORKER_ARCHITECTURE,
            "status":"SKIPPED_VERIFIED"
        })
        continue
    result=process_one_sample(row)
    session_results.append(result)

session_df=pd.DataFrame(session_results)

session_stamp=time.strftime("%Y%m%d_%H%M%S")
worker_tag = WORKER_NAME.replace(".", "_")
session_df.to_csv(EXPORT5/f"worker_{worker_tag}_session_{session_stamp}.csv",index=False)

verified_now=[sid for sid in WORKER_IDS if checkpoint_complete(sid)]
print(f"Worker {WORKER_NAME} session complete: {len(verified_now)}/20 verified")
if len(verified_now)<20:
    print(f"Remaining: {20-len(verified_now)}")
else:
    print(f"All 20 assigned {WORKER_ARCHITECTURE} checkpoints verified")

rows=[]
missing=[]
for sid in WORKER_IDS:
    d=checkpoint_dir(sid)
    if not checkpoint_complete(sid):
        missing.append(sid)
        continue
    rows.append(json.loads((d/"metadata.json").read_text()))

index=pd.DataFrame(rows)
print(f"Worker {WORKER_NAME} verified checkpoints:",len(index),"/20")

if missing:
    print("Missing/unverified assigned IDs:")
    for sid in missing:
        print(" -",sid)
    raise RuntimeError(
        f"Worker {WORKER_NAME} is incomplete. Resume production; do not freeze this worker."
    )

if set(index.sample_id.astype(str)) != set(WORKER_IDS):
    raise RuntimeError(f"Final checkpoint ID set does not equal the frozen {WORKER_NAME} assignment.")
if len(index)!=20 or index.sample_id.nunique()!=20:
    raise RuntimeError("Expected exactly 20 unique checkpoints.")
if set(index.architecture)!={WORKER_ARCHITECTURE}:
    raise RuntimeError(f"Wrong architecture checkpoint found in Worker {WORKER_NAME}.")
if not index.solver_completed.all() or not index.checkpoint_verified.all():
    raise RuntimeError("Solver/checkpoint verification gate failed.")
if not (index.ccx_returncode==0).all():
    raise RuntimeError("Nonzero CalculiX return code found.")
if not (index.mesh_tets<=MAX_TETS).all() or not (index.mesh_nodes<=MAX_NODES).all():
    raise RuntimeError("Mesh resource gate failure found.")
if not (index.mesh_q01>=MIN_Q01).all() or not (index.mesh_edge_ratio_p99<=MAX_EDGE_RATIO_P99).all():
    raise RuntimeError("Mesh quality gate failure found.")
if not (index.k32_radius_p95_mm<=MAX_K32_RADIUS_P95_MM).all():
    raise RuntimeError("Mapping-radius gate failure found.")
if not index.full_tensor_targets_authorized.eq(False).all():
    raise RuntimeError("Unauthorized tensor-target checkpoint found.")

order={sid:i for i,sid in enumerate(WORKER_IDS)}
index["_order"]=index.sample_id.astype(str).map(order)
index=index.sort_values("_order").drop(columns="_order").reset_index(drop=True)
index.to_csv(EXPORT5/f"{worker_tag}_FINAL_production_index.csv",index=False)

qc=pd.DataFrame([
    ["exactly 20 assigned checkpoints",len(index)==20],
    ["exact frozen ID set",set(index.sample_id.astype(str))==set(WORKER_IDS)],
    [f"all {WORKER_ARCHITECTURE}",(index.architecture==WORKER_ARCHITECTURE).all()],
    ["all solver return codes zero",(index.ccx_returncode==0).all()],
    ["all checkpoints verified",index.checkpoint_verified.all()],
    ["all mesh resource gates pass",((index.mesh_tets<=MAX_TETS)&(index.mesh_nodes<=MAX_NODES)).all()],
    ["all mesh quality gates pass",((index.mesh_q01>=MIN_Q01)&(index.mesh_edge_ratio_p99<=MAX_EDGE_RATIO_P99)).all()],
    ["all mapping gates pass",(index.k32_radius_p95_mm<=MAX_K32_RADIUS_P95_MM).all()],
    ["scalar targets only",index.full_tensor_targets_authorized.eq(False).all()],
],columns=["gate","pass"])
qc.to_csv(EXPORT5/f"{worker_tag}_FINAL_qc.csv",index=False)

freeze={
    "stage":WORKER_NAME,
    "worker":WORKER_NAME[-1],
    "status":"FROZEN",
    "architecture":WORKER_ARCHITECTURE,
    "assignment_rule":f"half {WORKER_HALF} of {WORKER_ARCHITECTURE} rows in authoritative manifest order",
    "sample_ids":WORKER_IDS,
    "n_samples":20,
    "mesh_protocol":{"label":"S250k",**PRODUCTION_MESH_CFG},
    "canonical_graph_resolution":"128^3",
    "mapping":{"k":32,"method":"FE-centroid inverse-distance weighting","power":2.0},
    "authorized_targets":AUTHORIZED_TARGETS,
    "full_tensor_targets_authorized":False,
    "primitive_closure_decision":PRIMITIVE_CLOSURE_DECISION,
    "all_final_qc_pass":bool(qc["pass"].all()),
}
(EXPORT5/f"{worker_tag}_FINAL_freeze.json").write_text(json.dumps(freeze,indent=2))

print(f"Worker {WORKER_NAME} final QC: {freeze['all_final_qc_pass']}")

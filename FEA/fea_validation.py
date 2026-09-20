#!/usr/bin/env python3
"""Validate the multi-architecture production geometry, graph, and mesh protocol.

Converted from `04_Multi_Architecture_Production_Protocol_Validation.ipynb`. Notebook prose and cell output were intentionally omitted.
"""
# MODULE 0 — Install/import dependencies (Colab-safe)
import os, sys, subprocess, importlib.util

def ensure(pkg, pip_name=None):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name or pkg])

for pkg,pipn in [('trimesh','trimesh'),('tetgen','tetgen'),('skimage','scikit-image'),('networkx','networkx'),('psutil','psutil')]:
    ensure(pkg,pipn)

import gc, re, json, time, shutil, hashlib, zipfile, ctypes
from pathlib import Path
import numpy as np, pandas as pd
import trimesh, tetgen, networkx as nx, psutil
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from skimage import measure
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

print('Python:',sys.version.split()[0])
print('RAM GB:',round(psutil.virtual_memory().total/1e9,2))
print('TetGen:',getattr(tetgen,'__version__','unknown'))

# MODULE 1 — Frozen protocol and folders
SEED=20260916
np.random.seed(SEED)

ROOT=Path.cwd()/"tpms_notebook_04"
INPUT=ROOT/"input"; PILOT=ROOT/"pilot"; EXPORT=ROOT/"exports"
for p in [INPUT,PILOT,EXPORT]: p.mkdir(parents=True,exist_ok=True)

ARCHS=['gyroid','diamond','primitive']
DOMAIN_MM=np.array([20.,20.,20.])
ORIGIN_MM=np.array([-10.,-10.,0.])
PROD_N=128
QC_RHO_ABS_ERR_MAX=0.010

CONNECTIVITY=26
SHORT_EDGE_THICKNESS_FRAC=0.35
SPUR_THICKNESS_FRAC=0.50

E_MPA=110000.0
NU=0.33
TARGET_MACRO_STRAIN=0.01
TOP_DISPLACEMENT_MM=-TARGET_MACRO_STRAIN*DOMAIN_MM[2]

BALANCED_CFG={'minratio':1.30,'mindihedral':10.0,'steinerleft':250000}
MAX_TETS_PREFLIGHT=3_000_000
MAX_NODES_PREFLIGHT=900_000
MIN_Q01=0.10
MAX_EDGE_RATIO_P99=8.0

# Default is safe: geometry+graph+mesh preflight only. Turn on only after reviewing preflight.
RUN_OPTIONAL_FEA=False

CCX=shutil.which('ccx')
print('ccx:',CCX)
print('RUN_OPTIONAL_FEA =',RUN_OPTIONAL_FEA)

# MODULE 2 — Upload/extract frozen Notebook-03.1 package
# Expected ZIP: 03_1_FINAL_TPMS_DATASET_DESIGN.zip
try:
    from google.colab import files
    uploaded=files.upload()
    for name,data in uploaded.items():
        p=INPUT/name; p.write_bytes(data)
except ImportError:
    print('Not Colab: place 03_1_FINAL_TPMS_DATASET_DESIGN.zip in',INPUT)

for z in INPUT.glob('*.zip'):
    with zipfile.ZipFile(z) as f: f.extractall(INPUT)

manifest_file=next(iter(INPUT.rglob('03_FINAL_population_120_WITH_SPLITS.csv')),None)
if manifest_file is None:
    raise RuntimeError('03_FINAL_population_120_WITH_SPLITS.csv not found. Upload the final Notebook-03.1 ZIP.')
manifest=pd.read_csv(manifest_file)
assert len(manifest)==120 and manifest.sample_id.is_unique
assert set(manifest.architecture)==set(ARCHS)
assert (manifest.groupby('architecture').size().reindex(ARCHS)==40).all()
print('Loaded authoritative manifest:',manifest_file)
display(manifest.groupby(['architecture','iid_split']).size().unstack(fill_value=0))

# MODULE 3 — Frozen implicit TPMS production geometry generator

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
    lo=0.; hi=float(np.quantile(absF,min(.95,max(.60,target+.30))))+1e-6
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

def hard_gc():
    gc.collect()
    try: ctypes.CDLL('libc.so.6').malloc_trim(0)
    except: pass

# MODULE 4 — Frozen canonical graph helper functions (from Notebook 01.2)
from collections import defaultdict, deque
from scipy import ndimage
from skimage.morphology import skeletonize
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
    # Current scikit-image skeletonize supports 3D binary arrays.
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
    # Junction voxels are degree > 2. Endpoints remain individual terminals.
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

    # Handle rare isolated skeleton voxels explicitly.
    for v in np.where(deg==0)[0]:
        sid=len(supernodes)
        supernodes.append({"kind":"isolated","voxels":[int(v)],"pos":world[v].copy()})
        voxel_to_super[int(v)]=sid

    return supernodes,voxel_to_super,deg

def trace_region_branches(coords, adj, world, supernodes, voxel_to_super, deg):
    # Boundary half-edges leaving any contracted region/end point.
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
                    # Degree-2 chain should have exactly one forward continuation.
                    if not candidates: break
                    # If numerical topology creates >1 continuation outside a declared junction,
                    # stop safely at the current voxel by promoting it later.
                    if len(candidates)>1:
                        break
                    nn=candidates[0]
                    visited.add(tuple(sorted((cur,nn))))
                    path.append(nn); prev,cur=cur,nn
                    end_sid=voxel_to_super.get(cur)

                if end_sid is None:
                    # Promote unresolved branch tip to a synthetic terminal.
                    end_sid=len(supernodes)
                    supernodes.append({"kind":"synthetic_terminal","voxels":[int(cur)],
                                       "pos":world[cur].copy()})
                    voxel_to_super[int(cur)]=end_sid

                if end_sid==sid:
                    # Closed micro-loop inside a junction region is not a useful structural branch.
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
    # Keep the physically shortest path for accidental parallel duplicates.
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
    # Union-find contraction of edges below a physical threshold.
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
        phi=float(np.arccos(np.clip(dv[2]/chord,-1,1))) if chord>0 else 0.
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

print("Robust skeleton/graph utilities loaded.")

# MODULE 5 — Canonical graph extraction directly from the 128³ production mask

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

# MODULE 6 — Deterministic six-sample pilot selection (low/high density per architecture)
pilot_rows=[]
for arch in ARCHS:
    d=manifest[manifest.architecture==arch].sort_values(['target_relative_density','sample_id'])
    pilot_rows += [d.iloc[0],d.iloc[-1]]
pilots=pd.DataFrame(pilot_rows).reset_index(drop=True)
pilots['pilot_role']=['low','high']*3
print('Six geometry-only pilot scaffolds:')
display(pilots[['sample_id','architecture','pilot_role','target_relative_density','cell_size_mm','grading_mode','grading_amplitude']])
pilots.to_csv(EXPORT/'04_pilot_selection.csv',index=False)

# MODULE 7 — Regenerate six pilots at 128³ + geometry QC + canonical graph QC
geom_rows=[]
for _,row in pilots.iterrows():
    sid=row.sample_id; print('\n',sid,row.architecture,row.pilot_role)
    t0=time.time(); mask,tau,rho,spacing=calibrate_mask(row,PROD_N)
    comps=int(ndimage.label(mask,structure=np.ones((3,3,3),np.uint8))[1])
    mesh=surface_from_mask(mask,spacing)
    nodes,edges,gq=analyze_mask_graph(mask,spacing)
    out={
      'sample_id':sid,'architecture':row.architecture,'pilot_role':row.pilot_role,
      'target_density':float(row.target_relative_density),'rho_128':rho,
      'rho_abs_error_128':abs(rho-float(row.target_relative_density)),'tau_128':tau,
      'solid_components_26':comps,'surface_vertices':len(mesh.vertices),'surface_faces':len(mesh.faces),
      'watertight':bool(mesh.is_watertight),'winding':bool(mesh.is_winding_consistent),
      'surface_components':len(mesh.split(only_watertight=False)),**gq,
    }
    out['geometry_pass']=bool(out['rho_abs_error_128']<=QC_RHO_ABS_ERR_MAX and comps==1 and out['watertight'] and out['winding'] and out['surface_components']==1)
    out['graph_pass']=bool(gq['graph_components']==1 and gq['graph_nodes']>20 and gq['graph_edges']>20)
    out['runtime_s']=time.time()-t0
    d=PILOT/sid; d.mkdir(exist_ok=True)
    mesh.export(d/'production_surface_128.stl')
    nodes.to_csv(d/'canonical_nodes.csv',index=False); edges.to_csv(d/'canonical_edges.csv',index=False)
    np.savez_compressed(d/'geometry_mask_128.npz',mask=mask,spacing=spacing,tau=tau)
    geom_rows.append(out)
    print({k:out[k] for k in ['rho_abs_error_128','solid_components_26','surface_faces','graph_nodes','graph_edges','geometry_pass','graph_pass']})
    del mask,mesh,nodes,edges; hard_gc()
geom_qc=pd.DataFrame(geom_rows); display(geom_qc)
geom_qc.to_csv(EXPORT/'04_geometry_graph_preflight.csv',index=False)
assert geom_qc.geometry_pass.all(), 'STOP: at least one 128³ production geometry failed.'
assert geom_qc.graph_pass.all(), 'STOP: at least one canonical graph failed.'
print('✓ All six pilots passed 128³ geometry + graph preflight.')

# MODULE 8 — Balanced TetGen mesh preflight helpers

def tet_volumes(p,t):
    P=p[t]
    return np.abs(np.einsum('ij,ij->i',P[:,1]-P[:,0],np.cross(P[:,2]-P[:,0],P[:,3]-P[:,0])))/6

def mesh_qc(p,t):
    P=p[t]; vol=tet_volumes(p,t)
    ee=np.stack([np.linalg.norm(P[:,0]-P[:,1],1),],axis=0) if False else np.stack([
      np.linalg.norm(P[:,0]-P[:,1],axis=1),np.linalg.norm(P[:,0]-P[:,2],axis=1),np.linalg.norm(P[:,0]-P[:,3],axis=1),
      np.linalg.norm(P[:,1]-P[:,2],axis=1),np.linalg.norm(P[:,1]-P[:,3],axis=1),np.linalg.norm(P[:,2]-P[:,3],axis=1)],axis=1)
    q=12*(3*vol)**(2/3)/np.maximum((ee**2).sum(1),1e-30)
    er=ee.max(1)/np.maximum(ee.min(1),1e-30)
    return vol,q,er

def exterior_faces(t):
    ff=np.vstack([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[1,2,3]],t[:,[2,0,3]]])
    ss=np.sort(ff,axis=1); order=np.lexsort((ss[:,2],ss[:,1],ss[:,0])); s=ss[order]
    same=np.all(s[1:]==s[:-1],axis=1); starts=np.r_[0,np.where(~same)[0]+1]; ends=np.r_[starts[1:],len(s)]
    return ff[order[starts[(ends-starts)==1]]]

def end_sets(p,t,pitch):
    bf=exterior_faces(t); tri=p[bf]; c=tri.mean(1)
    nv=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); nz=np.abs((nv/np.maximum(np.linalg.norm(nv,axis=1)[:,None],1e-15))[:,2])
    z0,z1=p[:,2].min(),p[:,2].max(); tol=1.5*pitch
    bot=np.unique(bf[(c[:,2]<=z0+tol)&(nz>=.80)]); top=np.unique(bf[(c[:,2]>=z1-tol)&(nz>=.80)])
    return bf,bot,top

# MODULE 9 — Balanced mesh preflight on all six pilots (NO FEA)
mesh_rows=[]
for _,row in pilots.iterrows():
    sid=row.sample_id; d=PILOT/sid; print('\nMeshing',sid)
    surf=trimesh.load_mesh(d/'production_surface_128.stl',process=False)
    tg=tetgen.TetGen(np.asarray(surf.vertices,np.float64),np.asarray(surf.faces,np.int32))
    t0=time.time()
    res=tg.tetrahedralize(order=1,quality=True,**BALANCED_CFG)
    sec=time.time()-t0
    p=np.asarray(res[0],np.float64); t=np.asarray(res[1],np.int64)
    vol,q,er=mesh_qc(p,t); pitch=DOMAIN_MM[0]/PROD_N
    bf,bot,top=end_sets(p,t,pitch)
    rowq={
      'sample_id':sid,'architecture':row.architecture,'pilot_role':row.pilot_role,
      'nodes':len(p),'tets':len(t),'mesh_s':sec,'zero_volume_tets':int((vol<=1e-14).sum()),
      'q_p01':float(np.quantile(q,.01)),'q_median':float(np.median(q)),'edge_ratio_p99':float(np.quantile(er,.99)),
      'exterior_faces':len(bf),'bottom_nodes':len(bot),'top_nodes':len(top),
    }
    rowq['resource_pass']=bool(len(t)<=MAX_TETS_PREFLIGHT and len(p)<=MAX_NODES_PREFLIGHT)
    rowq['quality_pass']=bool(rowq['zero_volume_tets']==0 and rowq['q_p01']>=MIN_Q01 and rowq['edge_ratio_p99']<=MAX_EDGE_RATIO_P99 and len(bot)>100 and len(top)>100)
    np.savez_compressed(d/'balanced_mesh.npz',points=p,tets=t,bottom=bot,top=top)
    mesh_rows.append(rowq); print(rowq)
    del surf,tg,res,p,t,vol,q,er,bf,bot,top; hard_gc()
mesh_preflight=pd.DataFrame(mesh_rows); display(mesh_preflight)
mesh_preflight.to_csv(EXPORT/'04_balanced_mesh_preflight.csv',index=False)
MESH_PROTOCOL_PREFLIGHT_PASS=bool(mesh_preflight.resource_pass.all() and mesh_preflight.quality_pass.all())
print('MESH_PROTOCOL_PREFLIGHT_PASS =',MESH_PROTOCOL_PREFLIGHT_PASS)
if not MESH_PROTOCOL_PREFLIGHT_PASS:
    print('STOP before FEA. Review failed rows; do not loosen gates automatically.')

# MODULE 10 — Select one median-density FEA anchor per architecture
anchors=[]
for arch in ARCHS:
    d=manifest[manifest.architecture==arch].copy()
    med=d.target_relative_density.median()
    anchors.append(d.iloc[np.argmin(np.abs(d.target_relative_density.to_numpy()-med))])
anchors=pd.DataFrame(anchors).reset_index(drop=True)
print('Optional FEA anchors (one per architecture):')
display(anchors[['sample_id','architecture','target_relative_density','cell_size_mm','grading_mode']])
anchors.to_csv(EXPORT/'04_optional_fea_anchors.csv',index=False)
print('\nNOTE: anchors may differ from the six low/high mesh pilots. RUN_OPTIONAL_FEA defaults to False.')

# MODULE 11 — Optional FEA gate
if RUN_OPTIONAL_FEA:
    if not MESH_PROTOCOL_PREFLIGHT_PASS:
        raise RuntimeError('Cannot run FEA: six-sample mesh preflight did not pass.')
    if CCX is None:
        raise RuntimeError('CalculiX ccx is not installed/found.')
    print('FEA requested. For Notebook 04, run one full-field anchor per architecture only after its 128³ geometry and balanced mesh are generated with the same helpers above.')
    print('This notebook intentionally does not auto-launch unseen anchor meshes/solves in the same cell; keep RUN_OPTIONAL_FEA=False for the first pass and send the preflight table for review.')
else:
    print('✓ Safe default: no CalculiX solve launched.')
    print('Send 04_balanced_mesh_preflight results for review before enabling any multi-hour FEA.')

# MODULE 12 — Authoritative Notebook-04 preflight QC + package
qc=pd.DataFrame({
 'gate':['six pilot geometries selected','all 128^3 geometry QC pass','all canonical graphs connected','six balanced meshes generated','balanced mesh resource gates pass','balanced mesh quality gates pass'],
 'pass':[len(pilots)==6,bool(geom_qc.geometry_pass.all()),bool(geom_qc.graph_pass.all()),len(mesh_preflight)==6,bool(mesh_preflight.resource_pass.all()),bool(mesh_preflight.quality_pass.all())]
})
display(qc)
NOTEBOOK04_PREFLIGHT_PASS=bool(qc['pass'].all())
summary={
 'notebook':'04','purpose':'multi-architecture production protocol validation before 120-sample FEA',
 'production_resolution':PROD_N,'balanced_tetgen':BALANCED_CFG,'pilot_count':6,
 'NOTEBOOK04_PREFLIGHT_PASS':NOTEBOOK04_PREFLIGHT_PASS,'RUN_OPTIONAL_FEA':RUN_OPTIONAL_FEA,
 'next_step':'If preflight passes, review resource envelope; then complete 3 architecture-anchor FEA validation before Notebook 05 full production.'
}
with open(EXPORT/'04_protocol_validation_summary.json','w') as f: json.dump(summary,f,indent=2)
qc.to_csv(EXPORT/'04_authoritative_preflight_qc.csv',index=False)

zip_path=Path.cwd()/'04_TPMS_PRODUCTION_PROTOCOL_PREFLIGHT.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in EXPORT.glob('*'): z.write(p,arcname=p.name)
print('NOTEBOOK04_PREFLIGHT_PASS =',NOTEBOOK04_PREFLIGHT_PASS)
print('Package:',zip_path)
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError: pass

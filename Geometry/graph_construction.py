#!/usr/bin/env python3
"""Construct canonical TPMS surface and skeleton graphs.

"""

import json, math, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import trimesh
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
import torch
from torch_geometric.data import Data

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

ROOT = Path.cwd() / 'tpms_graph_project_v1_1'
RAW, PROC, FIG = ROOT/'data'/'raw', ROOT/'data'/'processed', ROOT/'figures'
for p in (RAW, PROC, FIG): p.mkdir(parents=True, exist_ok=True)

GITHUB_API = 'https://api.github.com/repos/metudust/RegionTPMS/contents/STL_FileDemos'

def download_regiontpms_sample(out_dir=RAW):
    req=urllib.request.Request(GITHUB_API, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r: items=json.load(r)
    stls=[x for x in items if x.get('name','').lower().endswith('.stl')]
    if not stls: raise RuntimeError('No STL files found in RegionTPMS/STL_FileDemos')
    def rank(x):
        n=x['name'].lower(); return (0 if 'gyroid' in n else 1 if 'diamond' in n else 2, n)
    chosen=sorted(stls,key=rank)[0]
    target=out_dir/chosen['name']
    urllib.request.urlretrieve(chosen['download_url'], target)
    return target, {'source':'RegionTPMS','repository':'https://github.com/metudust/RegionTPMS',
                    'source_page':chosen['html_url'],'license':'MIT','original_name':chosen['name'],
                    'external_sample':True}

def generate_gyroid_fallback(path, n=72, cells=3, level=0.0):
    x=np.linspace(-np.pi*cells,np.pi*cells,n)
    X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
    f=np.sin(X)*np.cos(Y)+np.sin(Y)*np.cos(Z)+np.sin(Z)*np.cos(X)
    band=np.abs(f)-0.35
    verts,faces,_,_=measure.marching_cubes(band, level=level)
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=True)
    m.apply_scale(20.0/max(m.extents))
    m.export(path)
    return path, {'source':'deterministic local Gyroid-like fallback','license':'generated in stage',
                  'external_sample':False,'generator':{'n':n,'cells':cells,'band_threshold':0.35}}

try:
    stl_path, provenance = download_regiontpms_sample()
    print('Downloaded public sample:', stl_path.name)
except Exception as e:
    warnings.warn(f'Public download failed ({e}); using deterministic fallback.')
    stl_path, provenance = generate_gyroid_fallback(RAW/'fallback_gyroid.stl')

GEOMETRY_UNIT = 'STL_unit'  # Change only when source provenance establishes the physical unit.
loaded=trimesh.load(stl_path, force='mesh')
mesh=loaded.dump(concatenate=True) if isinstance(loaded,trimesh.Scene) else loaded
mesh=mesh.copy(); mesh.merge_vertices(); mesh.remove_unreferenced_vertices()
components=mesh.split(only_watertight=False)
areas=np.array([m.area for m in components]) if components else np.array([])
summary={
 'vertices':int(len(mesh.vertices)), 'faces':int(len(mesh.faces)),
 'watertight':bool(mesh.is_watertight), 'winding_consistent':bool(mesh.is_winding_consistent),
 'surface_area':float(mesh.area), 'volume_if_watertight':float(abs(mesh.volume)) if mesh.is_watertight else None,
 'bounds_min':mesh.bounds[0].tolist(), 'bounds_max':mesh.bounds[1].tolist(),
 'extents':mesh.extents.tolist(), 'components':int(len(components)),
 'largest_component_area_fraction':float(areas.max()/areas.sum()) if len(areas) else None,
 'geometry_unit':GEOMETRY_UNIT
}
if not mesh.is_watertight:
    warnings.warn('Mesh is not watertight. Filled-voxel thickness and later volumetric FEA may be unreliable.')
clean_stl=PROC/'scaffold_clean.stl'; mesh.export(clean_stl)
print('Saved cleaned STL:',clean_stl)

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
face_ids=np.arange(len(mesh.faces))
if len(face_ids)>18000: face_ids=rng.choice(face_ids,18000,replace=False)
tri=mesh.vertices[mesh.faces[face_ids]]
fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection='3d')
ax.add_collection3d(Poly3DCollection(tri,alpha=.28,linewidths=.05))
mins,maxs=mesh.bounds
ax.set(xlim=(mins[0],maxs[0]),ylim=(mins[1],maxs[1]),zlim=(mins[2],maxs[2]),
       xlabel='X',ylabel='Y',zlabel='Z',title='Input TPMS surface mesh')
plt.tight_layout(); plt.show()

V=np.asarray(mesh.vertices,float); F=np.asarray(mesh.faces,int)
pairs=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); pairs=np.sort(pairs,axis=1); E=np.unique(pairs,axis=0)
Gm=nx.Graph(); Gm.add_nodes_from(range(len(V))); Gm.add_edges_from(map(tuple,E))
deg=np.array([Gm.degree(i) for i in range(len(V))],float)
mins,maxs=V.min(0),V.max(0); span=np.maximum(maxs-mins,1e-12); pos_norm=(V-mins)/span
vec=V[E[:,1]]-V[E[:,0]]; length=np.linalg.norm(vec,axis=1); safe=np.maximum(length,1e-12)
median_edge=float(np.median(length))
boundary_tol=float(min(median_edge,0.01*span[2]))
bottom=(V[:,2] <= mins[2]+boundary_tol).astype(np.float32)
top=(V[:,2] >= maxs[2]-boundary_tol).astype(np.float32)
X_mesh=np.c_[pos_norm,deg/np.maximum(deg.max(),1),bottom,top]
theta=np.arccos(np.clip(vec[:,2]/safe,-1,1)); phi=np.arctan2(vec[:,1],vec[:,0])
edge_attr_mesh=np.c_[length,theta,phi]

nodes_mesh=pd.DataFrame({'node_id':np.arange(len(V)),'x':V[:,0],'y':V[:,1],'z':V[:,2],
 'x_norm':pos_norm[:,0],'y_norm':pos_norm[:,1],'z_norm':pos_norm[:,2],
 'degree':deg.astype(int),'is_bottom':bottom.astype(int),'is_top':top.astype(int)})
edges_mesh=pd.DataFrame({'edge_id':np.arange(len(E)),'source':E[:,0],'target':E[:,1],
 'length':length,'theta':theta,'phi':phi})
qc_mesh={'num_nodes':len(V),'num_edges':len(E),'connected_components':nx.number_connected_components(Gm),
 'isolated_nodes':len(list(nx.isolates(Gm))),'mean_degree':float(deg.mean()),
 'min_edge_length':float(length.min()),'median_edge_length':median_edge,'max_edge_length':float(length.max()),
 'boundary_tolerance':boundary_tol,'bottom_nodes':int(bottom.sum()),'top_nodes':int(top.sum()),
 'finite_edge_features':bool(np.isfinite(edge_attr_mesh).all())}

fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection='3d')
ids=np.arange(len(E));
if len(ids)>12000: ids=rng.choice(ids,12000,replace=False)
for a,b in E[ids]:
    p,q=V[a],V[b]; ax.plot([p[0],q[0]],[p[1],q[1]],[p[2],q[2]],linewidth=.25,alpha=.30)
ax.set(xlabel='X',ylabel='Y',zlabel='Z',title='Representation A — surface mesh graph (sampled)')
plt.tight_layout(); plt.show()

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


RESOLUTIONS=[64,128,256]
CONNECTIVITY=26
SHORT_EDGE_THICKNESS_FRAC=0.35
SPUR_THICKNESS_FRAC=0.50

def analyze_and_simplify(mesh,target_resolution,connectivity=26):
    solid,T,pitch=voxelize_filled(mesh,target_resolution)
    skel=skeletonize_3d_compat(solid)
    coords,adj=skel_graph(skel,connectivity)
    world=ijk_to_world(coords,T)
    edt=ndimage.distance_transform_edt(solid)

    supernodes,v2s,deg=contract_junction_regions(coords,adj,world)
    supernodes,branches=trace_region_branches(coords,adj,world,supernodes,v2s,deg)
    nodes=node_dataframe(supernodes)
    edges=branch_dataframe(supernodes,branches,edt,coords,pitch)
    nodes,edges=dedupe_edges(nodes,edges)
    nodes,edges=reindex_graph(nodes,edges)

    median_t=float(np.nanmedian(edges.local_thickness_mean)) if len(edges) else np.nan
    contract_thr=SHORT_EDGE_THICKNESS_FRAC*median_t
    spur_thr=SPUR_THICKNESS_FRAC*median_t

    raw_nodes,raw_edges=nodes.copy(),edges.copy()
    nodes,edges=contract_short_edges(nodes,edges,contract_thr)
    nodes,edges=prune_terminal_spurs(nodes,edges,spur_thr)
    nodes,edges=dedupe_edges(nodes,edges)
    nodes,edges=reindex_graph(nodes,edges)
    edges=recompute_edge_geometry(nodes,edges)

    degf=graph_degrees(nodes,edges)
    nodes["degree"]=degf

    summary={
        "target_resolution":target_resolution,
        "pitch":pitch,
        "solid_voxels":int(solid.sum()),
        "skeleton_voxels":int(skel.sum()),
        "raw_branch_nodes":len(raw_nodes),
        "raw_branch_edges":len(raw_edges),
        "simplified_nodes":len(nodes),
        "simplified_edges":len(edges),
        "components":graph_components(nodes,edges),
        "total_path_length":float(edges.path_length.sum()) if len(edges) else 0.0,
        "median_path_length":float(edges.path_length.median()) if len(edges) else np.nan,
        "mean_tortuosity":float(edges.tortuosity.mean()) if len(edges) else np.nan,
        "median_local_thickness":median_t,
        "contract_threshold":contract_thr,
        "spur_threshold":spur_thr,
        "endpoint_count":int((nodes.degree==1).sum()) if len(nodes) else 0,
        "junction_count":int((nodes.degree>=3).sum()) if len(nodes) else 0,
    }
    return summary,{
        "solid":solid,"T":T,"pitch":pitch,"skel":skel,"coords":coords,
        "raw_nodes":raw_nodes,"raw_edges":raw_edges,
        "nodes":nodes,"edges":edges,
        "spectrum":spectral_signature(nodes,edges)
    }

summaries=[]; resolution_results={}
for n in RESOLUTIONS:
    print(f"Analyzing {n}^3-equivalent resolution ...")
    s,r=analyze_and_simplify(mesh,n,CONNECTIVITY)
    summaries.append(s); resolution_results[n]=r

convergence=pd.DataFrame(summaries)

def pct_change(a,b):
    return 100.0*abs(b-a)/max(abs(a),1e-12)

metrics=["simplified_nodes","simplified_edges","total_path_length","median_path_length",
         "mean_tortuosity","median_local_thickness","endpoint_count","junction_count"]
rows=[]
for a,b in zip(RESOLUTIONS[:-1],RESOLUTIONS[1:]):
    ra=convergence.set_index("target_resolution").loc[a]
    rb=convergence.set_index("target_resolution").loc[b]
    row={"comparison":f"{a}->{b}"}
    for m in metrics: row[m+"_pct"]=pct_change(float(ra[m]),float(rb[m]))
    rows.append(row)
conv_pct=pd.DataFrame(rows)

q=[0.1,0.25,0.5,0.75,0.9]
distrows=[]
for n in RESOLUTIONS:
    e=resolution_results[n]["edges"]
    rr={"resolution":n}
    for qq,val in zip(q,e.path_length.quantile(q).to_numpy()):
        rr[f"L_q{int(qq*100)}"]=float(val)
    distrows.append(rr)
length_quantiles=pd.DataFrame(distrows)

spectral_rows=[]
for n in RESOLUTIONS:
    vals=resolution_results[n]["spectrum"]
    row={"resolution":n}
    for i,v in enumerate(vals[:10],1): row[f"lambda_{i}"]=float(v)
    spectral_rows.append(row)
spectral_df=pd.DataFrame(spectral_rows)

fig,ax=plt.subplots(figsize=(7,4))
ax.plot(convergence.target_resolution,convergence.simplified_nodes,marker="o",label="nodes")
ax.plot(convergence.target_resolution,convergence.simplified_edges,marker="o",label="edges")
ax.set_xlabel("Target resolution"); ax.set_ylabel("Count"); ax.set_title("Simplified graph convergence")
ax.legend(); plt.show()

fig,ax=plt.subplots(figsize=(7,4))
ax.plot(convergence.target_resolution,convergence.total_path_length,marker="o")
ax.set_xlabel("Target resolution"); ax.set_ylabel("Total path length")
ax.set_title("Geometric convergence"); plt.show()

connectivity_rows=[]
for conn in [6,18,26]:
    solid,T,pitch=voxelize_filled(mesh,128)
    skel=skeletonize_3d_compat(solid)
    coords,adj=skel_graph(skel,conn)
    ii=[]; jj=[]
    for i,a in enumerate(adj):
        for j in a:
            ii.append(i); jj.append(j)
    if len(coords):
        A=csr_matrix((np.ones(len(ii)),(ii,jj)),shape=(len(coords),len(coords)))
        comps=int(connected_components(A,directed=False,return_labels=False))
    else: comps=0
    connectivity_rows.append({"connectivity":conn,"skeleton_voxels":len(coords),
                              "raw_skeleton_components":comps})
connectivity_df=pd.DataFrame(connectivity_rows)

CANONICAL_RESOLUTION = 128
SENSITIVITY_RESOLUTION = 256

canonical = resolution_results[CANONICAL_RESOLUTION]
nodes_struct = canonical["nodes"].copy()
edges_struct = canonical["edges"].copy()

surface_edge_vectors = V[E[:,0]] - V[E[:,1]]
surface_edge_lengths = np.linalg.norm(surface_edge_vectors,axis=1)
edge_scale = float(np.median(surface_edge_lengths))

zmin,zmax = float(V[:,2].min()),float(V[:,2].max())
boundary_tol = max(2.0*edge_scale,1e-6*(zmax-zmin))

nodes_struct["bottom_flag"] = (nodes_struct.z <= zmin+boundary_tol).astype(int)
nodes_struct["top_flag"] = (nodes_struct.z >= zmax-boundary_tol).astype(int)

nodes_struct["dist_to_bottom"] = nodes_struct.z - zmin
nodes_struct["dist_to_top"] = zmax - nodes_struct.z

xyz=nodes_struct[["x","y","z"]].to_numpy(float)
lo=V.min(axis=0); hi=V.max(axis=0); span=np.maximum(hi-lo,1e-12)
norm=(xyz-lo)/span
nodes_struct[["x_norm","y_norm","z_norm"]] = norm
nodes_struct["degree_norm"] = nodes_struct.degree/np.maximum(nodes_struct.degree.max(),1)


fig=plt.figure(figsize=(9,8))
ax=fig.add_subplot(111,projection="3d")
P=nodes_struct[["x","y","z"]].to_numpy()
for s,t in edges_struct[["source","target"]].to_numpy(int):
    q=P[[s,t]]
    ax.plot(q[:,0],q[:,1],q[:,2],linewidth=.5,alpha=.45)
ax.scatter(P[:,0],P[:,1],P[:,2],s=7)
ax.set_title(f"Canonical simplified structural graph — {CANONICAL_RESOLUTION}³")
plt.show()

c128=convergence.set_index("target_resolution").loc[128]
c256=convergence.set_index("target_resolution").loc[256]

def pc(a,b):
    return 100.0*abs(float(b)-float(a))/max(abs(float(a)),1e-12)

required_node=["x_norm","y_norm","z_norm","degree_norm","top_flag","bottom_flag",
               "dist_to_bottom","dist_to_top"]
required_edge=["path_length","chord_length","tortuosity","theta","phi",
               "local_thickness_mean","local_thickness_min","voxel_count"]

qc = {
    "canonical_connected": int(c128.components)==1,
    "sensitivity_connected": int(c256.components)==1,
    "junctions_converged_10pct": pc(c128.junction_count,c256.junction_count)<=10,
    "edges_converged_25pct": pc(c128.simplified_edges,c256.simplified_edges)<=25,
    "path_length_converged_15pct": pc(c128.total_path_length,c256.total_path_length)<=15,
    "thickness_converged_15pct": pc(c128.median_local_thickness,c256.median_local_thickness)<=15,
    "tortuosity_converged_10pct": pc(c128.mean_tortuosity,c256.mean_tortuosity)<=10,
    "node_features_finite": np.isfinite(nodes_struct[required_node].to_numpy(float)).all(),
    "edge_features_finite": np.isfinite(edges_struct[required_edge].to_numpy(float)).all(),
    "no_self_loops": bool((edges_struct.source!=edges_struct.target).all()),
}

diagnostics = {
    "total_nodes_change_pct_128_to_256": pc(c128.simplified_nodes,c256.simplified_nodes),
    "endpoint_change_pct_128_to_256": pc(c128.endpoint_count,c256.endpoint_count),
    "junction_change_pct_128_to_256": pc(c128.junction_count,c256.junction_count),
    "edge_change_pct_128_to_256": pc(c128.simplified_edges,c256.simplified_edges),
    "path_length_change_pct_128_to_256": pc(c128.total_path_length,c256.total_path_length),
    "thickness_change_pct_128_to_256": pc(c128.median_local_thickness,c256.median_local_thickness),
    "tortuosity_change_pct_128_to_256": pc(c128.mean_tortuosity,c256.mean_tortuosity),
    "canonical_bottom_flag_count": int(nodes_struct.bottom_flag.sum()),
    "canonical_top_flag_count": int(nodes_struct.top_flag.sum()),
    "canonical_min_dist_bottom": float(nodes_struct.dist_to_bottom.min()),
    "canonical_min_dist_top": float(nodes_struct.dist_to_top.min()),
}

qc_df=pd.DataFrame({"gate":list(qc.keys()),
                    "status":["PASS" if bool(v) else "WARN" for v in qc.values()]})
diag_df=pd.DataFrame({"diagnostic":list(diagnostics.keys()),
                      "value":list(diagnostics.values())})
READY_FOR_FEA=all(bool(v) for v in qc.values())
print("READY_FOR_FEA =",READY_FOR_FEA)
if not READY_FOR_FEA:
    print("Inspect WARN gates before proceeding.")

surface_xyz=V.astype(float)
surface_lo=surface_xyz.min(axis=0); surface_hi=surface_xyz.max(axis=0)
surface_span=np.maximum(surface_hi-surface_lo,1e-12)
surface_xyz_norm=(surface_xyz-surface_lo)/surface_span
nodes_mesh["x_norm"]=surface_xyz_norm[:,0]
nodes_mesh["y_norm"]=surface_xyz_norm[:,1]
nodes_mesh["z_norm"]=surface_xyz_norm[:,2]

surface_degree=np.zeros(len(V),dtype=int)
for u,v in E:
    surface_degree[u]+=1; surface_degree[v]+=1
nodes_mesh["degree"]=surface_degree
nodes_mesh["degree_norm"]=surface_degree/max(surface_degree.max(),1)

nodes_mesh["bottom_flag"]=(V[:,2] <= zmin+boundary_tol).astype(int)
nodes_mesh["top_flag"]=(V[:,2] >= zmax-boundary_tol).astype(int)
nodes_mesh["dist_to_bottom"]=V[:,2]-zmin
nodes_mesh["dist_to_top"]=zmax-V[:,2]

surface_node_features=["x_norm","y_norm","z_norm","degree_norm","top_flag","bottom_flag",
                       "dist_to_bottom","dist_to_top"]

surface_bidir=np.vstack([E,E[:,::-1]])
surface_attr_bidir=np.vstack([edge_attr_mesh,edge_attr_mesh])

assert np.isfinite(nodes_mesh[surface_node_features].to_numpy(float)).all()
assert np.isfinite(surface_attr_bidir).all()

surface_data=Data(
    x=torch.tensor(nodes_mesh[surface_node_features].to_numpy(float),dtype=torch.float32),
    edge_index=torch.tensor(surface_bidir.T,dtype=torch.long),
    edge_attr=torch.tensor(surface_attr_bidir,dtype=torch.float32),
    pos=torch.tensor(V,dtype=torch.float32)
)

struct_node_features=required_node
struct_edge_features=required_edge

se=edges_struct[["source","target"]].to_numpy(int)
se_bidir=np.vstack([se,se[:,::-1]])
sa=edges_struct[struct_edge_features].to_numpy(float)
sa_bidir=np.vstack([sa,sa])

assert np.isfinite(nodes_struct[struct_node_features].to_numpy(float)).all()
assert np.isfinite(sa_bidir).all()

struct_data=Data(
    x=torch.tensor(nodes_struct[struct_node_features].to_numpy(float),dtype=torch.float32),
    edge_index=torch.tensor(se_bidir.T,dtype=torch.long),
    edge_attr=torch.tensor(sa_bidir,dtype=torch.float32),
    pos=torch.tensor(nodes_struct[["x","y","z"]].to_numpy(float),dtype=torch.float32)
)


nodes_mesh.to_csv(PROC/"mesh_nodes.csv",index=False)
edges_mesh.to_csv(PROC/"mesh_edges.csv",index=False)
nodes_struct.to_csv(PROC/"canonical_structural_nodes_128.csv",index=False)
edges_struct.to_csv(PROC/"canonical_structural_edges_128.csv",index=False)
convergence.to_csv(PROC/"resolution_convergence_simplified.csv",index=False)
conv_pct.to_csv(PROC/"resolution_convergence_percent.csv",index=False)
length_quantiles.to_csv(PROC/"branch_length_quantiles.csv",index=False)
spectral_df.to_csv(PROC/"laplacian_spectrum.csv",index=False)
connectivity_df.to_csv(PROC/"connectivity_sensitivity.csv",index=False)
qc_df.to_csv(PROC/"pre_fea_qc_final.csv",index=False)
diag_df.to_csv(PROC/"resolution_selection_diagnostics.csv",index=False)

torch.save(surface_data,PROC/"surface_graph.pt")
torch.save(struct_data,PROC/"canonical_structural_graph_128.pt")
mesh.export(PROC/"scaffold_clean.stl")

metadata={
    "stage_version":"01.2-final",
    "source_file":str(stl_path),
    "geometry_unit":GEOMETRY_UNIT,
    "mesh_vertices":int(len(V)),
    "mesh_faces":int(len(F)),
    "mesh_watertight":bool(mesh.is_watertight),
    "tested_resolutions":RESOLUTIONS,
    "canonical_structural_resolution":CANONICAL_RESOLUTION,
    "sensitivity_resolution":SENSITIVITY_RESOLUTION,
    "connectivity":CONNECTIVITY,
    "short_edge_thickness_fraction":SHORT_EDGE_THICKNESS_FRAC,
    "spur_thickness_fraction":SPUR_THICKNESS_FRAC,
    "boundary_tolerance":float(boundary_tol),
    "ready_for_fea":bool(READY_FOR_FEA),
    "qc":{k:bool(v) for k,v in qc.items()},
    "resolution_diagnostics":{k:float(v) for k,v in diagnostics.items()},
    "resolution_selection_rationale":
        "128^3 selected from geometry-only convergence before FEA labels. "
        "64->128 is highly stable; 256 preserves junction/branch geometry but adds terminal detail "
        "at substantially higher voxel cost.",
    "boundary_note":
        "Binary top/bottom flags are diagnostics. Continuous distance-to-face features are retained "
        "because the medial graph is interior and need not intersect physical loading faces.",
    "representation_note":
        "Medial graph is a compact topology-preserving geometric representation, not literal TPMS struts."
}
(PROC/"metadata_01_2_final.json").write_text(json.dumps(metadata,indent=2))

print(f"Graph construction complete: outputs saved to {PROC}")

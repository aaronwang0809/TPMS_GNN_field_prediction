#!/usr/bin/env python3
"""Train and compare the published Maurizi encode-process-decode GNN variants."""
import os,sys,json,time,random,subprocess
from pathlib import Path
import numpy as np, pandas as pd
SEED=2026
def seed_all(s=SEED):
 random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
 try:
  import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
 except: pass
seed_all()
ROOT=Path(os.environ.get("TPMS_PROJECT_ROOT",Path.cwd()/"TPMS_IEEE_BIGDATA")).expanduser().resolve(); N06=ROOT/"Stage06"; N07=ROOT/"Stage07"
OUT=ROOT/"Stage07_1"; CK=OUT/"checkpoints"; PR=OUT/"predictions"
for p in [OUT,CK,PR]: p.mkdir(parents=True,exist_ok=True)
req=[N06/"06_FINALIZED.txt",N06/"06_final_ml_dataset_index.csv",N06/"06_normalization_train_only.json",N07/"07_FINALIZED.txt",N07/"07_final_test_metrics.csv",N07/"07_test_metrics_per_graph.csv"]
miss=[str(p) for p in req if not p.is_file()]
if miss: raise RuntimeError("Missing frozen handoff:\n"+"\n".join(miss))
idx=pd.read_csv(N06/"06_final_ml_dataset_index.csv"); idx["sample_id"]=idx.sample_id.astype(str); idx["iid_split"]=idx.iid_split.astype(str).str.lower(); idx["architecture"]=idx.architecture.astype(str).str.lower()
assert len(idx)==110 and idx.sample_id.is_unique
assert idx.iid_split.value_counts().to_dict()=={"train":70,"test":21,"validation":19}
assert all(Path(p).is_file() for p in idx.graph_file.astype(str))
norm=json.loads((N06/"06_normalization_train_only.json").read_text())
import torch
try: import torch_geometric
except:
 subprocess.check_call([sys.executable,"-m","pip","install","-q","torch-geometric"]); import torch_geometric
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type!="cuda": raise RuntimeError("GPU required — keep A100 runtime")
print("GPU:",torch.cuda.get_device_name(0))

from torch_geometric.data import Data
MU=np.asarray(norm["target_log_mean"],float); SD=np.asarray(norm["target_log_std"],float); TARGETS=["von_mises_stress","equivalent_strain"]
def inverse_y(y): return np.expm1(np.asarray(y,float)*SD+MU)
def load(row,physical=False):
 with np.load(str(row.graph_file),allow_pickle=False) as z:
  d=Data(x=torch.from_numpy(z["x"]).float(),edge_index=torch.from_numpy(z["edge_index"]).long(),edge_attr=torch.from_numpy(z["edge_attr"]).float(),y=torch.from_numpy(z["y"]).float()); d.graph_attr=torch.from_numpy(z["graph_attr"]).float().reshape(1,-1)
  if physical: d.y_phys=torch.from_numpy(z["y_phys"]).float()
 d.sample_id=str(row.sample_id); d.architecture=str(row.architecture); return d
tr=idx[idx.iid_split=="train"].reset_index(drop=True); va=idx[idx.iid_split=="validation"].reset_index(drop=True)
d=load(tr.iloc[0]); assert d.x.shape[1]==8 and d.edge_attr.shape[1]==9 and d.graph_attr.shape==(1,12) and d.y.shape[1]==2

import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import MessagePassing
def MLP(i,h): return nn.Sequential(nn.Linear(i,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.LayerNorm(h))
class MauriziBlock(MessagePassing):
 def __init__(self,h): super().__init__(aggr="add"); self.edge_net=MLP(3*h,h); self.node_net=MLP(2*h,h)
 def forward(self,x,ei,e):
  nx=self.propagate(ei,x=x,edge_attr=e); row,col=ei; ne=self.edge_net(torch.cat([x[row],x[col],e],-1)); return x+nx,e+ne
 def message(self,x_i,x_j,edge_attr): return self.edge_net(torch.cat([x_i,x_j,edge_attr],-1))
 def update(self,aggr_out,x): return self.node_net(torch.cat([aggr_out,x],-1))
class MauriziEPD(nn.Module):
 def __init__(self,h=16,steps=15):
  super().__init__(); self.ne=MLP(20,h); self.ee=MLP(9,h); self.block=MauriziBlock(h); self.dec=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2)); self.steps=steps
 def forward(self,d):
  x=self.ne(torch.cat([d.x,d.graph_attr.expand(d.x.size(0),-1)],-1)); e=self.ee(d.edge_attr)
  for _ in range(self.steps): x,e=self.block(x,d.edge_index,e)
  return self.dec(x)
CONFIGS={"MauriziPaperConfig":{"h":16,"steps":15},"MauriziMatchedTraining":{"h":128,"steps":4}}
def gate(name,cfg,kind,lr):
 seed_all(); m=MauriziEPD(**cfg).to(DEVICE); d=load(tr.iloc[0]).to(DEVICE); o=m(d); L=F.l1_loss(o,d.y) if kind=="l1" else F.mse_loss(o,d.y); assert o.shape==d.y.shape and torch.isfinite(L); L.backward()
 tiny=[load(tr.iloc[i]).to(DEVICE) for i in range(2)]; m=MauriziEPD(**cfg).to(DEVICE); op=torch.optim.Adam(m.parameters(),lr=lr)
 def ev():
  m.eval(); a=[]
  with torch.no_grad():
   for q in tiny:
    z=m(q); a.append((F.l1_loss(z,q.y) if kind=="l1" else F.mse_loss(z,q.y)).item())
  return np.mean(a)
 a=ev()
 for _ in range(160):
  m.train(); op.zero_grad(); losses=[]
  for q in tiny:
   z=m(q); losses.append(F.l1_loss(z,q.y) if kind=="l1" else F.mse_loss(z,q.y))
  torch.stack(losses).mean().backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); op.step()
 b=ev()
 if not np.isfinite(b) or b>=.60*a: raise RuntimeError(name+" tiny-overfit FAILED")
 del m,tiny; torch.cuda.empty_cache()
gate("MauriziPaperConfig",CONFIGS["MauriziPaperConfig"],"l1",.01); gate("MauriziMatchedTraining",CONFIGS["MauriziMatchedTraining"],"mse",.003)

@torch.no_grad()
def eval_loss(m,rows,kind):
 m.eval(); s=0.0; n=0
 for _,r in rows.iterrows():
  d=load(r).to(DEVICE); e=m(d)-d.y; s+=(torch.abs(e).sum() if kind=="l1" else torch.square(e).sum()).item(); n+=e.numel()
 return s/n
def train(name,protocol):
 seed_all(); cfg=CONFIGS[name]; m=MauriziEPD(**cfg).to(DEVICE)
 if protocol=="paper": kind="l1"; opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=5e-4); sch=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.9); accum=1; patience=45
 else: kind="mse"; opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=10,min_lr=1e-5); accum=4; patience=35
 rng=np.random.default_rng(SEED); best=1e99; be=-1; bad=0; hist=[]; t0=time.time(); path=CK/f"{name}_best.pt"
 for ep in range(1,301):
  m.train(); opt.zero_grad(set_to_none=True); ss=0.0; nn_=0; ac=0; order=rng.permutation(len(tr))
  for j,i in enumerate(order,1):
   d=load(tr.iloc[i]).to(DEVICE); o=m(d); L=F.l1_loss(o,d.y) if kind=="l1" else F.mse_loss(o,d.y); (L/accum).backward(); ac+=1; e=o.detach()-d.y; ss+=(torch.abs(e).sum() if kind=="l1" else torch.square(e).sum()).item(); nn_+=e.numel()
   if ac==accum or j==len(order): torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); opt.zero_grad(set_to_none=True); ac=0
  vl=eval_loss(m,va,kind); tl=ss/nn_; sch.step() if protocol=="paper" else sch.step(vl); lr=opt.param_groups[0]["lr"]; hist.append({"epoch":ep,"train_loss":tl,"val_loss":vl,"lr":lr})
  if vl<best-1e-4: best=vl; be=ep; bad=0; torch.save({"state_dict":m.state_dict(),"cfg":cfg,"protocol":protocol,"best_epoch":be,"best_val":best},path)
  else: bad+=1
  if ep==1 or ep%10==0 or bad==0: print(f"{name:23s} ep {ep:3d} train {tl:.5f} val {vl:.5f} best {best:.5f}@{be} lr {lr:.2e}")
  if bad>=patience: print(name,"early stop",ep); break
 pd.DataFrame(hist).to_csv(OUT/f"{name}_history.csv",index=False); mins=(time.time()-t0)/60; del m; torch.cuda.empty_cache(); return {"model":name,"protocol":protocol,"best_val_loss":best,"best_epoch":be,"train_minutes":mins,"checkpoint":str(path)}

res=[]; res.append(train("MauriziPaperConfig","paper")); res.append(train("MauriziMatchedTraining","matched")); vr=pd.DataFrame(res); vr.to_csv(OUT/"07_1_maurizi_validation_results.csv",index=False)
freeze={"frozen_before_test":True,"seed":SEED,"variants":res,"paper_config":{"latent":16,"message_steps":15,"aggregation":"sum","loss":"MAE","optimizer":"Adam","lr":.01,"weight_decay":5e-4,"gamma":.9},"matched_training":{"latent":128,"message_steps":4,"loss":"MSE","optimizer":"AdamW","lr":1e-3,"weight_decay":1e-5}}
(OUT/"07_1_MODEL_CONFIGS_FROZEN_BEFORE_TEST.json").write_text(json.dumps(freeze,indent=2))

te=idx[idx.iid_split=="test"].reset_index(drop=True); assert len(te)==21
def metrics(y,p):
 y=np.asarray(y,float); p=np.asarray(p,float); e=p-y; rmse=float(np.sqrt(np.mean(e*e))); mae=float(np.mean(abs(e))); rg=float(y.max()-y.min()); sst=float(np.sum((y-y.mean())**2)); r2=1-float(np.sum(e*e))/sst if sst>0 else np.nan; rr=float(np.corrcoef(y,p)[0,1]) if y.std()>0 and p.std()>0 else np.nan; return {"RMSE":rmse,"MAE":mae,"NRMSE_range":rmse/rg if rg>0 else np.nan,"R2":r2,"Pearson_r":rr}
rows=[]; pgr=[]
for name,cfg in CONFIGS.items():
 pay=torch.load(CK/f"{name}_best.pt",map_location=DEVICE); m=MauriziEPD(**cfg).to(DEVICE); m.load_state_dict(pay["state_dict"]); m.eval(); YY=[]; PP=[]; tt=[]
 with torch.no_grad():
  for _,r in te.iterrows():
   d=load(r,True).to(DEVICE); torch.cuda.synchronize(); t=time.perf_counter(); pn=m(d); torch.cuda.synchronize(); ms=(time.perf_counter()-t)*1000; pp=inverse_y(pn.cpu().numpy()); yy=d.y_phys.cpu().numpy().astype(float); YY.append(yy); PP.append(pp); tt.append(ms); rec={"model":name,"sample_id":r.sample_id,"architecture":r.architecture,"num_nodes":len(yy),"inference_ms":ms}
   for k,tg in enumerate(TARGETS):
    for mk,mv in metrics(yy[:,k],pp[:,k]).items(): rec[f"{tg}_{mk}"]=mv
   pgr.append(rec); np.savez_compressed(PR/f"{r.sample_id}_{name}.npz",y_true_phys=yy.astype("float32"),y_pred_phys=pp.astype("float32"))
 Y=np.concatenate(YY); P=np.concatenate(PP)
 for k,tg in enumerate(TARGETS): rows.append({"model":name,"target":tg,**metrics(Y[:,k],P[:,k]),"mean_inference_ms_per_graph":float(np.mean(tt)),"median_inference_ms_per_graph":float(np.median(tt))})
 del m; torch.cuda.empty_cache()
lit=pd.DataFrame(rows); pg=pd.DataFrame(pgr); lit.to_csv(OUT/"07_1_maurizi_test_metrics.csv",index=False); pg.to_csv(OUT/"07_1_maurizi_test_metrics_per_graph.csv",index=False)

old=pd.read_csv(N07/"07_final_test_metrics.csv"); full=pd.concat([old,lit],ignore_index=True); order=["NodeMLP","GraphSAGE","MauriziPaperConfig","MauriziMatchedTraining","EdgeAwareGNN"]; full["ord"]=full.model.map({m:i for i,m in enumerate(order)}); full=full.sort_values(["target","ord"]).drop(columns="ord").reset_index(drop=True); full.to_csv(OUT/"07_1_FULL_MODEL_COMPARISON.csv",index=False)
imp=[]
for tg in TARGETS:
 for met in ["RMSE","MAE"]:
  b=float(full[(full.model=="MauriziPaperConfig")&(full.target==tg)][met].iloc[0]); p=float(full[(full.model=="EdgeAwareGNN")&(full.target==tg)][met].iloc[0]); imp.append({"target":tg,"metric":met,"MauriziPaperConfig":b,"EdgeAwareGNN":p,"EdgeAware_relative_error_reduction_percent":100*(b-p)/b})
imp=pd.DataFrame(imp); imp.to_csv(OUT/"07_1_edgeaware_vs_maurizi_improvement.csv",index=False)

oldpg=pd.read_csv(N07/"07_test_metrics_per_graph.csv"); comb=pd.concat([oldpg[oldpg.model=="EdgeAwareGNN"],pg],ignore_index=True); ar=[]
for (mo,a),g in comb.groupby(["model","architecture"]):
 z={"model":mo,"architecture":a,"n_graphs":len(g),"inference_ms_mean":float(g.inference_ms.mean())}
 for tg in TARGETS:
  for met in ["RMSE","MAE","NRMSE_range","R2","Pearson_r"]: z[f"{tg}_{met}_mean_per_graph"]=float(g[f"{tg}_{met}"].mean())
 ar.append(z)
arch=pd.DataFrame(ar).sort_values(["architecture","model"]); arch.to_csv(OUT/"07_1_architecturewise_edgeaware_vs_maurizi.csv",index=False)
paper=full[["model","target","RMSE","MAE","NRMSE_range","R2","Pearson_r","mean_inference_ms_per_graph"]].copy(); paper["NRMSE_percent"]=100*paper.pop("NRMSE_range"); paper.to_csv(OUT/"07_1_PAPER_READY_MODEL_TABLE.csv",index=False)

required=[OUT/"07_1_maurizi_validation_results.csv",OUT/"07_1_MODEL_CONFIGS_FROZEN_BEFORE_TEST.json",OUT/"07_1_maurizi_test_metrics.csv",OUT/"07_1_FULL_MODEL_COMPARISON.csv",OUT/"07_1_edgeaware_vs_maurizi_improvement.csv",OUT/"07_1_architecturewise_edgeaware_vs_maurizi.csv",OUT/"07_1_PAPER_READY_MODEL_TABLE.csv"]
for p in required:
 if not p.is_file(): raise RuntimeError("Missing "+str(p))
summary={"reference":"Maurizi, Gao & Berto, Scientific Reports 12, 21834 (2022)","official_repository":"https://github.com/marcomau06/GNNs_fields_prediction","comparison_type":"published architecture benchmark","dataset_n":110,"split":{"train":70,"validation":19,"test":21},"models":["NodeMLP","GraphSAGE","MauriziPaperConfig","MauriziMatchedTraining","EdgeAwareGNN"],"seed":SEED}
print("Benchmark complete: 5-model comparison finalized")

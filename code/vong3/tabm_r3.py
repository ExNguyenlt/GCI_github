# Vong 3 (A1 vong 1 #5): TabM-mini (Gorishniy et al. 2024, goi tabm 0.0.3) + piecewise-linear embeddings, torch CPU.
# PYTHONPATH=pylib_torch:pylib_tabm python3 tabm_r3.py <tag> [fold,...]   -> r3m/<tag>_f<f>.npz ; du 5 fold -> oof_<tag>.npy / pt_<tag>.npy
# Hop le: chuan hoa/bin chi tu dac trung (khong nhan); early stopping tren 10% tach ra TU PHAN TRAIN cua fold (khong dung nhan fold cham).
import sys, os, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
import torch, torch.nn as nn, tabm, rtdl_num_embeddings as rne
from fe_r3 import load_r3
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score as A
tag=sys.argv[1]; FL=[int(f) for f in sys.argv[2].split(',')] if len(sys.argv)>2 else [0,1,2,3,4]
K=int(os.environ.get('K','8')); DB=int(os.environ.get('DB','192')); DE=int(os.environ.get('DE','4')); MAXEP=int(os.environ.get('EP','30')); BS=1024; SD=100*int(os.environ.get('SEED','0'))
torch.set_num_threads(int(os.environ.get('NT','4')))
df,y,n,cats,ids,G=load_r3()
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; cols=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']+G['PQ']
num=[c for c in cols if c not in cats]; D=df[num].astype(float).replace([np.inf,-np.inf],np.nan)
ind=[c for c in num if D[c].isna().mean()>0.01]; M=D[ind].isna().astype(np.float32).values
Q=QuantileTransformer(n_quantiles=1000,output_distribution='normal',random_state=0).fit_transform(D.values); Q=np.nan_to_num(Q,nan=0.0).astype(np.float32)
XN=np.hstack([Q,M]); XC=np.column_stack([pd.factorize(df[c].astype(str))[0] for c in cats]).astype(np.int64); card=[int(XC[:,j].max())+1 for j in range(XC.shape[1])]
print('num',XN.shape,'cat',XC.shape,flush=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y)); yt=torch.tensor(y,dtype=torch.float32)
XNt=torch.tensor(XN); XCt=torch.tensor(XC)
def pred(m,idx):
    m.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(idx),8192):
            j=idx[i:i+8192]; out.append(torch.sigmoid(m(XNt[j],XCt[j]).squeeze(-1)).mean(1))
    return torch.cat(out).numpy()
for f in FL:
    fo=f'{S}/r3m/{tag}_f{f}.npz'
    if os.path.exists(fo): continue
    a,b=folds[f]; tr,va=train_test_split(a,test_size=0.1,stratify=y[a],random_state=f+SD); t0=time.time(); torch.manual_seed(f+SD)
    bins=[]
    for j in range(len(num)):
        e=np.unique(np.quantile(XN[tr,j],np.linspace(0,1,17))); e=e if len(e)>=2 else np.array([e[0]-1,e[0]+1]); bins.append(torch.tensor(e,dtype=torch.float32))
    bins+=[torch.tensor([-0.5,0.5,1.5]) for _ in ind]
    m=tabm.TabM.make(n_num_features=XN.shape[1],cat_cardinalities=card,d_out=1,num_embeddings=rne.PiecewiseLinearEmbeddings(bins,d_embedding=DE,activation=False,version='B'),k=K,n_blocks=2,d_block=DB,dropout=0.15,arch_type='tabm-mini')
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=3e-4); best=-1; bad=0; state=None; trt=torch.tensor(tr)
    for ep in range(MAXEP):
        m.train(); perm=trt[torch.randperm(len(trt))]
        for i in range(0,len(perm)-BS+1,BS):
            j=perm[i:i+BS]; out=m(XNt[j],XCt[j]).squeeze(-1); loss=nn.functional.binary_cross_entropy_with_logits(out,yt[j][:,None].expand(-1,K))
            opt.zero_grad(); loss.backward(); opt.step()
        av=A(y[va],pred(m,va)); print(tag,'fold',f,'ep',ep,'inner-val AUC',round(av,5),f'{time.time()-t0:.0f}s',flush=True)
        if av>best: best=av; bad=0; state={k:v.clone() for k,v in m.state_dict().items()}
        else:
            bad+=1
            if bad>=4: break
    m.load_state_dict(state); pb=pred(m,b); ptst=pred(m,np.arange(n,len(df)))
    print(tag,'fold',f,'AUC',round(A(y[b],pb),5),'inner best',round(best,5),f'{time.time()-t0:.0f}s',flush=True); np.savez(fo,pb=pb,pt=ptst)
if all(os.path.exists(f'{S}/r3m/{tag}_f{f}.npz') for f in range(5)):
    oof=np.zeros(n); pt=np.zeros(len(df)-n)
    for f,(a,b) in enumerate(folds): z=np.load(f'{S}/r3m/{tag}_f{f}.npz'); oof[b]=z['pb']; pt+=z['pt']/5
    print(tag,'OOF AUC',round(A(y,oof),5),'folds',[round(A(y[b],oof[b]),5) for a,b in folds],flush=True); np.save(f'{S}/oof_{tag}.npy',oof); np.save(f'{S}/pt_{tag}.npy',pt)

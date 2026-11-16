# Vong 3: lrbin (LR tren one-hot bin 16 quantile) voi PROF + PQ. python3 -I div_r3.py lrbin lrbin_r3
#  lrbin: LR tren one-hot bin quantile 16 bin (nhu vong 1) + nhom PROF.   ebm: Explainable Boosting Machine (interpret-core 0.7.8 trong pylib).
import sys, os, time, numpy as np, pandas as pd, scipy.sparse as sp, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
which,tag=sys.argv[1],sys.argv[2]; NF=int(sys.argv[3]) if len(sys.argv)>3 else 5
from fe_r3 import load_r3; df,y,n,cats,ids,G=load_r3()
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc and c not in cats]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
CK=f'{S}/ck_{tag}.npz'; oof=np.zeros(n); pt=np.zeros(len(df)-n); done=0
if os.path.exists(CK): z=np.load(CK); oof,pt,done=z['oof'],z['pt'],int(z['done'])
t0=time.time()
if which=='lrbin':
    from sklearn.linear_model import LogisticRegression
    num=orig+G['IMP']+G['RATE']+G['TERM']+G['DATE']+G['PROF']+G['PQ']+(G['PROFX']+G['PROFR'] if 'x' in tag else []); NB=16; blocks=[]
    for c in num:
        x=df[c].values.astype(float); ok=~np.isnan(x)
        if np.unique(x[ok]).size<=NB: codes=pd.factorize(np.where(ok,x,-9e9))[0]
        else:
            qs=np.unique(np.nanquantile(x,np.linspace(0,1,NB+1)[1:-1])); codes=np.searchsorted(qs,x); codes[~ok]=len(qs)+1
        blocks.append(pd.get_dummies(codes.astype(str),prefix=c,sparse=False).astype(np.float32).values)
    for c in cats: blocks.append(pd.get_dummies(df[c].astype(str)).astype(np.float32).values)
    M=sp.csr_matrix(np.hstack(blocks)); print('design',M.shape,flush=True); Xtr,Xte=M[:n],M[n:]
else:
    sys.path.insert(0,S+'/pylib')
    from interpret.glassbox import ExplainableBoostingClassifier
    num=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']; D=df[num+list(cats)].copy()
    for c in cats: D[c]=D[c].astype(str)
    Xtr,Xte=D.iloc[:n].reset_index(drop=True),D.iloc[n:].reset_index(drop=True); print('design',D.shape,flush=True)
for f,(a,b) in enumerate(folds[:NF]):
    if f<done: continue
    if which=='lrbin':
        m=LogisticRegression(C=0.05,solver='lbfgs',max_iter=2000).fit(Xtr[a],y[a]); p=m.predict_proba(Xtr[b])[:,1]; q=m.predict_proba(Xte)[:,1]
    else:
        m=ExplainableBoostingClassifier(interactions=int(os.environ.get('EBM_INT','20')),max_bins=256,outer_bags=4,n_jobs=4,random_state=0).fit(Xtr.iloc[a],y[a])
        p=m.predict_proba(Xtr.iloc[b])[:,1]; q=m.predict_proba(Xte)[:,1]
    oof[b]=p; pt+=q/NF; print(tag,f,round(A(y[b],p),5),f'{time.time()-t0:.0f}s',flush=True); np.savez(CK,oof=oof,pt=pt,done=f+1)
if NF==5:
    print(tag,'OOF AUC',round(A(y,oof),5),'folds',[round(A(y[b],oof[b]),5) for a,b in folds],flush=True)
    np.save(f'{S}/oof_{tag}.npy',oof); np.save(f'{S}/pt_{tag}.npy',pt)

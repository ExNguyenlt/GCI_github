# python3 -I ens_r1.py name1,name2,...  (ten file oof_<name>.npy / pt_<name>.npy)  [write]
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'
y=np.load(S+'/y.npy'); n=len(y); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def stack(names):
    O=np.column_stack([lg(np.load(f'{S}/oof_{m}.npy')) for m in names]); P=np.column_stack([lg(np.load(f'{S}/pt_{m}.npy')) for m in names])
    mu,sd=O.mean(0),O.std(0); O=(O-mu)/sd; P=(P-mu)/sd
    oof=np.zeros(n)
    for a,b in folds: oof[b]=LogisticRegression(C=1.0,max_iter=1000).fit(O[a],y[a]).decision_function(O[b])
    m=LogisticRegression(C=1.0,max_iter=1000).fit(O,y); return oof,m.decision_function(P),m.coef_.ravel()
def rankavg(names):
    O=np.mean([rankdata(np.load(f'{S}/oof_{m}.npy'))/n for m in names],0)
    P=np.mean([rankdata(np.load(f'{S}/pt_{m}.npy'))/61500 for m in names],0); return O,P
def fa(o): return np.array([A(y[b],o[b]) for a,b in folds])
R0=['lgb','cat','xgb','lgb_opt','cat_opt','xgb_opt']
base,_,_=stack(R0); fb=fa(base)
if __name__=='__main__':
    names=sys.argv[1].split(',')
    for m in names: o=np.load(f'{S}/oof_{m}.npy'); print(f'{m:16s} AUC {A(y,o):.5f} spearman vs base-stack {spearmanr(o,base)[0]:.4f}')
    so,sp,co=stack(names); ro,rp=rankavg(names)
    for t,o in [('stackLR',so),('rankavg',ro)]:
        d=fa(o)-fb; print(f'{t:8s} OOF {A(y,o):.5f} folds {np.round(fa(o),5)} diff vs r0-stack(0.76273) {np.round(d,5)} pos {(d>0).sum()}/5')
    print('stack coefs',dict(zip(names,np.round(co,3))))
    if len(sys.argv)>2:
        out=sys.argv[2]; P=sp if 'stack' in out else rp; np.save(f'{S}/pt_ens_final.npy',P); print('saved')

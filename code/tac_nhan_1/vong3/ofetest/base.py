# Khung sang loc A1 vong 3: BASE = REF + PROF (giong r2_screen '+PROF'), LGB lr 0.05, fold 0/1 x seed 0/1
import sys, os, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]
BASE=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=int(os.environ.get('NJ','1')))
JS=S+'/loop/a1r3/scr.json'
def load(): return json.load(open(JS)) if os.path.exists(JS) else {}
def fit(Xa,ya,Xb,yb,s,w=None,P2=None,dsp=None):
    p=dict(P,seed=s); p.update(P2 or {})
    return lgb.train(p,lgb.Dataset(Xa,ya,weight=w,params=dsp or {}),5000,valid_sets=[lgb.Dataset(Xb,yb,params=dsp or {})],callbacks=[lgb.early_stopping(200,verbose=False)])
def report(name,res,ref='BASE'):
    R=load(); R[name]=[float(x) for x in res]; json.dump(R,open(JS,'w'))
    msg=f'{name:34s} {np.round(res,5)} mean {np.mean(res):.5f}'
    if ref in R and ref!=name and len(R[ref])==len(res):
        d=np.array(res)-np.array(R[ref]); msg+=f' | vs {ref} {d.mean():+.5f} pos {int((d>0).sum())}/{len(d)}'
    print(msg,flush=True)
def screen(name,extra=None,P2=None,drop=(),ref='BASE',FS=(0,1),SS=(0,1),hook=None):
    """extra: DataFrame (len(df)) cot target-free them vao; hook(a,b,Xa,Xb)->(Xa,Xb,ya,w) cho thu nghiem dung nhan trong fold"""
    D=df[BASE].copy() if extra is None else pd.concat([df[BASE],extra.set_axis(df.index)],axis=1)
    D=D[[c for c in D.columns if c not in drop]]; X=D.iloc[:n].reset_index(drop=True); res=[]; t=time.time()
    for fo in FS:
        a,b=folds[fo]; Xa=X.iloc[a].reset_index(drop=True); Xb=X.iloc[b].reset_index(drop=True); ya=y[a]; w=None
        for s in SS:
            if hook: Xa2,Xb2,ya2,w2=hook(a,b,Xa,Xb,s)
            else: Xa2,Xb2,ya2,w2=Xa,Xb,ya,w
            m=fit(Xa2,ya2,Xb2,y[b],s,w2,P2); res.append(A(y[b],m.predict(Xb2,num_iteration=m.best_iteration)))
    report(name,res,ref); print(f'   ({time.time()-t:.0f}s, {D.shape[1]} cols)',flush=True)

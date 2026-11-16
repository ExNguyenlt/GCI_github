# sang loc nhanh nhu R3 (fold0/1 x 2 seed, lr 0.05, cay nong). Dung: screen.py S variant1 variant2 ...
# variant: ten trong VARS ben duoi; luon chay REF truoc va bao chenh ghep cap
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S=sys.argv[1]; sys.path.insert(0,S); sys.path.insert(0,S+'/loop/a1r2')
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1()
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); ref=[c for c in df.columns if c not in newc]+G['IMP']+G['RATE']+G['TERM']
EXTRA={}
def add(name,frame):
    for c in frame.columns: df[c]=frame[c].values
    EXTRA[name]=list(frame.columns)
import os
for f in sys.argv[2:]:
    nm,path=f.split('=')
    fr=pd.read_pickle(path); add(nm,fr)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
def run(cols,tag):
    X=df.iloc[:n][cols].reset_index(drop=True); res=[]
    for f in (0,1):
        a,b=folds[f]
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(X.iloc[a],y[a]),5000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(X.iloc[b],num_iteration=m.best_iteration)))
    print(f'{tag:34s}',np.round(res,5),'mean',round(np.mean(res),5),flush=True); return np.array(res)
r0=run(ref,'REF IMP+RATE+TERM')
for nm,cols in EXTRA.items():
    for mode in ['add','replTERM']:
        if mode=='replTERM': continue
        cc=ref+cols if mode=='add' else [c for c in ref if c not in G['TERM']]+cols+['PRODUCT_FREQ','CG_FREQ']
        r=run(cc,f'{mode} {nm}'); d=r-r0; print(f'   diff {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/4',flush=True)

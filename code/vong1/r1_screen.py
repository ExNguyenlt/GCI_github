import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, time; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1()
for c in cats: df[c]=df[c].astype('category')
X=df.iloc[:n].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
newc=set(sum(G.values(),[])); orig=[c for c in X.columns if c not in newc]
def run(cols,tag):
    res=[]; t=time.time()
    for f in (0,1):
        a,b=folds[f]
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(X.loc[a,cols],y[a]),5000,valid_sets=[lgb.Dataset(X.loc[b,cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(X.loc[b,cols],num_iteration=m.best_iteration)))
    print(f'{tag:28s}',np.round(res,5),'mean',round(np.mean(res),5),f'{time.time()-t:.0f}s',flush=True); return np.array(res)
R={}
R['BASE']=run(orig,'BASE')
def cmp(k,ref): d=R[k]-R[ref]; print(f'   {k} - {ref}: {np.round(d,5)} mean {d.mean():+.5f} pos {int((d>0).sum())}/4',flush=True)
for k,cols in [('IMP',G['IMP']),('RATE',G['RATE']),('IMP+RATE',G['IMP']+G['RATE'])]:
    R[k]=run(orig+cols,k); cmp(k,'BASE')
ref=orig+G['IMP']+G['RATE']
R['IR+TERM']=run(ref+G['TERM'],'IR+TERM'); cmp('IR+TERM','IMP+RATE')
R['IR+DATE']=run(ref+G['DATE'],'IR+DATE'); cmp('IR+DATE','IMP+RATE')
R['IR+TERM+DATE']=run(ref+G['TERM']+G['DATE'],'IR+TERM+DATE'); cmp('IR+TERM+DATE','IMP+RATE')
# RATE mo rong: tat ca ky han
np.save(S+'/r1_screen.npy',R,allow_pickle=True)

# sang loc KNN (long fold) tren bo dac trung tham chieu: python3 -I r1_screen2.py IMP,RATE[,..]
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, time; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1(); KN=pd.read_pickle(S+'/r1_knn.pkl')
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); cols=[c for c in df.columns if c not in newc]+sum([G[g] for g in sys.argv[1].split(',')],[])
X=df.iloc[:n][cols].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
def run(kc,tag):
    res=[]
    for f in (0,1):
        a,b=folds[f]; Xa=X.iloc[a].reset_index(drop=True); Xb=X.iloc[b].reset_index(drop=True)
        if kc: Xa=pd.concat([Xa,KN[f]['tr'][kc]],axis=1); Xb=pd.concat([Xb,KN[f]['va'][kc]],axis=1)
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(Xa,y[a]),5000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(Xb,num_iteration=m.best_iteration)))
    print(f'{tag:28s}',np.round(res,5),'mean',round(np.mean(res),5),flush=True); return np.array(res)
r0=run([],'REF '+sys.argv[1])
for kc,t in [(['KNN1_500','KNN2_500'],'+KNN500'),(['KNN1_500','KNN2_500','KNN1_200','KNN2_200','KNN1_50','KNN2_50'],'+KNN all')]:
    r=run(kc,t); d=r-r0; print(f'   diff {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/4',flush=True)

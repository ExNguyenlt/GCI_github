# python3 -I run_r1.py <lgb|cat|xgb> <featspec e.g. IMP,RATE,TERM> <useknn 0/1> <seeds e.g. 0,1> <outtag>
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
which,spec,useknn,seeds,tag=sys.argv[1],sys.argv[2],int(sys.argv[3]),[int(s) for s in sys.argv[4].split(',')],sys.argv[5]
df,y,n,cats,ids,G=load_r1()
newc=set(sum(G.values(),[])); cols=[c for c in df.columns if c not in newc]+sum([G[g] for g in spec.split(',') if g],[])
KN=pd.read_pickle(S+'/r1_knn.pkl') if useknn else None
KC=['KNN1_500','KNN2_500','KNN1_200','KNN2_200','KNN1_50','KNN2_50']
X=df.iloc[:n][cols].reset_index(drop=True); T=df.iloc[n:][cols].reset_index(drop=True)
if which in ('lgb','xgb'):
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
oof=np.zeros(n); pt=np.zeros(len(T)); t0=time.time()
for f,(a,b) in enumerate(folds):
    Xa,Xb,Tt=X.iloc[a].reset_index(drop=True),X.iloc[b].reset_index(drop=True),T.copy()
    if useknn:
        k=KN[f]; Xa=pd.concat([Xa,k['tr'][KC]],axis=1); Xb=pd.concat([Xb,k['va'][KC]],axis=1); Tt=pd.concat([Tt,k['te'][KC]],axis=1)
    pb=np.zeros(len(b)); ptf=np.zeros(len(T))
    for s in seeds:
        if which=='lgb':
            import lightgbm as lgb
            P=dict(objective='binary',learning_rate=0.01,num_leaves=16,max_depth=6,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l1=2,lambda_l2=10,min_gain_to_split=0.01,cat_smooth=50,cat_l2=20,max_cat_to_onehot=4,verbose=-1,n_jobs=4,seed=s)
            m=lgb.train(P,lgb.Dataset(Xa,y[a]),10000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
            p=m.predict(Xb,num_iteration=m.best_iteration); q=m.predict(Tt,num_iteration=m.best_iteration); it=m.best_iteration
        elif which=='cat':
            from catboost import CatBoostClassifier
            m=CatBoostClassifier(iterations=5000,learning_rate=0.03,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,thread_count=4,verbose=0,random_seed=s)
            m.fit(Xa,y[a],cat_features=cats,eval_set=(Xb,y[b]),use_best_model=True)
            p=m.predict_proba(Xb)[:,1]; q=m.predict_proba(Tt)[:,1]; it=m.best_iteration_
        else:
            import xgboost as xgb
            m=xgb.XGBClassifier(n_estimators=10000,learning_rate=0.01,max_depth=5,min_child_weight=30,subsample=0.8,colsample_bytree=0.3,reg_lambda=10,reg_alpha=2,tree_method='hist',enable_categorical=True,max_cat_to_onehot=4,eval_metric='auc',early_stopping_rounds=300,n_jobs=4,random_state=s)
            m.fit(Xa,y[a],eval_set=[(Xb,y[b])],verbose=False)
            p=m.predict_proba(Xb)[:,1]; q=m.predict_proba(Tt)[:,1]; it=m.best_iteration
        pb+=p/len(seeds); ptf+=q/len(seeds)
        print(tag,'fold',f,'seed',s,round(roc_auc_score(y[b],p),5),it,f'{time.time()-t0:.0f}s',flush=True)
    oof[b]=pb; pt+=ptf/5
    print(tag,'fold',f,'AUC',round(roc_auc_score(y[b],pb),5),flush=True)
print(tag,'OOF AUC',round(roc_auc_score(y,oof),5),'folds',[round(roc_auc_score(y[b],oof[b]),5) for a,b in folds],flush=True)
np.save(f'{S}/oof_{tag}_r1.npy',oof); np.save(f'{S}/pt_{tag}_r1.npy',pt)

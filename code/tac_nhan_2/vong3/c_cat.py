# Sang loc CatBoost (lr 0.1, fold 0/1, seed 0) — bien the tham so tren PROF+PCAT. Moc: catscr_pcat 0.76694 / 0.76959
import sys, os, time, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import PRODUCT_KEY
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; cols=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']
cats=list(cats); PROD=PRODUCT_KEY(df); vc=pd.Series(PROD).value_counts(); df['PRODUCT_CAT']=np.where(pd.Series(PROD).map(vc).values<30,'rare',PROD)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
V={'ctr1':dict(max_ctr_complexity=1),'depth4':dict(depth=4),'ordered':dict(boosting_type='Ordered'),'l2_20':dict(l2_leaf_reg=20),'nopcat_ref':{},'depth8':dict(depth=8),
   'ctr2':dict(max_ctr_complexity=2),'rs0':dict(random_strength=0.2),'bt1':dict(bootstrap_type='Bernoulli',subsample=0.7),'rsm03':dict(rsm=0.3)}
for v in sys.argv[1:]:
    kw=dict(iterations=5000,learning_rate=0.1,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,thread_count=4,verbose=0,random_seed=0); kw.update(V[v])
    cc=cols+['PRODUCT_CAT']; ct=cats+['PRODUCT_CAT']; X=df.iloc[:n][cc].reset_index(drop=True); res=[]; t=time.time()
    for f in (0,1):
        a,b=folds[f]; m=CatBoostClassifier(**kw); m.fit(X.iloc[a],y[a],cat_features=ct,eval_set=(X.iloc[b],y[b]),use_best_model=True)
        res.append(round(roc_auc_score(y[b],m.predict_proba(X.iloc[b])[:,1]),5)); print(v,'fold',f,res[-1],m.best_iteration_,f'{time.time()-t:.0f}s',flush=True)
    d=np.array(res)-np.array([0.76694,0.76959]); print(v,res,'mean',round(np.mean(res),5),'vs PCAT ref',np.round(d,5),'mean %+.5f'%d.mean(),flush=True)

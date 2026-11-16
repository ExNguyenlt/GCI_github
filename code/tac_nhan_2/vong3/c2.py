# (A) do lac quan cua early-stopping tren fold cham; (B) vi sao PCAT hai LGB; (C) vai bien the LGB re
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import PRODUCT_KEY, nested_te
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; COLS=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']
PROD=PRODUCT_KEY(df); vc=pd.Series(PROD).value_counts(); df['PRODUCT_CAT']=pd.Series(np.where(pd.Series(PROD).map(vc).values<30,'rare',PROD)).astype('category').values
X=df.iloc[:n].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=2,metric='auc')
def fit(cols,f,s,extra={},te=False,rec=None,trainauc=False):
    a,b=folds[f]; Xa=X.loc[a,cols].reset_index(drop=True); Xb=X.loc[b,cols].reset_index(drop=True)
    if te: ta,(tb,)=nested_te(PROD[:n],y,a,[b],10); Xa['TE_PRODUCT']=ta; Xb['TE_PRODUCT']=tb
    cb=[lgb.early_stopping(200,verbose=False)]
    if rec is not None: cb.append(lgb.record_evaluation(rec))
    m=lgb.train(dict(P,seed=s,**extra),lgb.Dataset(Xa,y[a]),5000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=cb)
    r=A(y[b],m.predict(Xb,num_iteration=m.best_iteration))
    if trainauc: return r,m.best_iteration,A(y[a],m.predict(Xa,num_iteration=m.best_iteration))
    return r,m.best_iteration
what=sys.argv[1]
if what=='A':
    for te in (False,True):
        cur=[];bi=[]
        for f in range(5):
            rec={}; r,it=fit(COLS,f,0,te=te,rec=rec); cur.append(np.array(rec['valid_0']['auc'])); bi.append(it)
        es=np.array([c.max() for c in cur]); 
        fx=[]
        for f in range(5):
            k=int(np.mean([bi[j] for j in range(5) if j!=f])); k=min(k,len(cur[f])); fx.append(cur[f][k-1])
        fx=np.array(fx); print('done fold curves',flush=True); print('TE',te,'best_iter',bi,'\n  AUC early-stop tren fold cham',np.round(es,5),'mean',round(es.mean(),5),'\n  AUC tai so vong = TB best_iter 4 fold khac',np.round(fx,5),'mean',round(fx.mean(),5),'\n  do lac quan ES',np.round(es-fx,5),'mean %+.5f'%(es-fx).mean(),flush=True)
if what=='B':
    V={'PROF':(COLS,{}),'PROF+PCAT':(COLS+['PRODUCT_CAT'],{}),
       'PROF+PCAT reg':(COLS+['PRODUCT_CAT'],dict(cat_smooth=300,cat_l2=100,min_data_per_group=300,max_cat_threshold=8)),
       'PROF+PCAT reg2':(COLS+['PRODUCT_CAT'],dict(cat_smooth=1000,cat_l2=300,min_data_per_group=1000,max_cat_threshold=4))}
    ref=None
    for k,(c,e) in V.items():
        res=[];its=[];tr=[]
        for f in (0,1):
            for s in (0,1):
                r,it,ta=fit(c,f,s,e,trainauc=True); res.append(r); its.append(it); tr.append(ta)
        res=np.array(res)
        if ref is None: ref=res
        print(f'{k:16s}',np.round(res,5),'mean',round(res.mean(),5),'vs PROF %+.5f pos %d/4'%((res-ref).mean(),(res-ref>0).sum()),'best_iter',its,'train AUC',np.round(np.mean(tr),4),'gap',np.round(np.mean(tr)-res.mean(),4),flush=True)
if what=='C':
    V={'PROF':({},False),'extra_trees':(dict(extra_trees=True),False),'ff0.4':(dict(feature_fraction=0.4),False),'leaves24_d6':(dict(num_leaves=24,max_depth=6,min_child_samples=150),False),
       'path_smooth20':(dict(path_smooth=20),False),'goss':(dict(data_sample_strategy='goss',bagging_freq=0,bagging_fraction=1.0),False),'linear_tree':(dict(linear_tree=True,linear_lambda=10),False)}
    ref=None
    for k in sys.argv[2:] if len(sys.argv)>2 else V:
        e,te=V[k]; res=[];its=[]; t=time.time()
        for f in (0,1):
            for s in (0,1):
                r,it=fit(COLS,f,s,e,te=te); res.append(r); its.append(it)
        res=np.array(res)
        if ref is None: ref=res
        print(f'{k:16s}',np.round(res,5),'mean',round(res.mean(),5),'vs PROF %+.5f pos %d/4'%((res-ref).mean(),(res-ref>0).sum()),'best_iter',its,f'{time.time()-t:.0f}s',flush=True)

# python3 -I run_r2.py <lgb|cat|xgb> <spec vd. +PROF-TERM_HAT[~TE][~PCAT]> <seeds vd. 0,1,2> <tag>
# 5-fold StratifiedKFold rs=42 day du; luu tam theo fold vao ck_<tag>.npz (chay lai se tiep tuc); ket qua oof_<tag>.npy / pt_<tag>.npy
# ~TE  : them cot TE PRODUCT long fold (train a: inner 5-fold trong a; valid b va test: tu toan a) — R14
# ~PCAT: them PRODUCT (round(C/A,2), muc < 30 dong gop lai) lam bien phan loai (CatBoost ordered TS)
import sys, os, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import load_r2, PRODUCT_KEY, nested_te
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
which,spec,seeds,tag=sys.argv[1],sys.argv[2],[int(s) for s in sys.argv[3].split(',')],sys.argv[4]
NF=int(sys.argv[5]) if len(sys.argv)>5 else 5      # so fold chay (2 = sang loc fold 0/1)
PK=S+'/r2_df.pkl'
if os.path.exists(PK): df,y,n,cats,ids,G=pd.read_pickle(PK)
else: r=load_r2(); pd.to_pickle(r,PK); df,y,n,cats,ids,G=r
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; cols=orig+G['IMP']+G['RATE']+G['TERM']
G['RES']=[e+'_RES' for e in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
parts=spec.split('~'); flags=parts[1:]
for sgn,nm in re.findall(r'([+-])([A-Za-z0-9_]+)',parts[0]):
    cc=G[nm] if nm in G else [nm]
    if sgn=='+': cols+=[c for c in cc if c not in cols]
    else: cols=[c for c in cols if c not in cc]
cats=list(cats); PROD=PRODUCT_KEY(df)
if 'PCAT' in flags:
    vc=pd.Series(PROD).value_counts(); df['PRODUCT_CAT']=np.where(pd.Series(PROD).map(vc).values<30,'rare',PROD); cols.append('PRODUCT_CAT'); cats.append('PRODUCT_CAT')
def addcat(name,key):
    k=pd.Series(key).astype(str); vc=k.value_counts(); df[name]=np.where(k.map(vc).values<30,'rare',k.values); cols.append(name); cats.append(name)
if 'KTR' in flags: addcat('TR_CAT',np.where((df.NAME_CONTRACT_TYPE=='Cash loans').values,df.TERM_HAT.values*100000+np.round(df.RATE_HAT.values*200),-5))
if 'KCG' in flags: addcat('CG_CAT',np.round(df.AMT_CREDIT.values/df.AMT_GOODS_PRICE.values,3))
if 'KANN' in flags: addcat('ANN_CAT',df.AMT_ANNUITY.values)
if 'KCR' in flags: addcat('CR_CAT',df.AMT_CREDIT.values)
X=df.iloc[:n][cols].reset_index(drop=True); T=df.iloc[n:][cols].reset_index(drop=True)
if which in ('lgb','xgb'):
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
CK=f'{S}/ck_{tag}.npz'; oof=np.zeros(n); pt=np.zeros(len(T)); done=0
if os.path.exists(CK): z=np.load(CK); oof,pt,done=z['oof'],z['pt'],int(z['done']); print(tag,'resume from fold',done,flush=True)
print(tag,which,spec,'cols',len(cols),'seeds',seeds,flush=True); t0=time.time()
for f,(a,b) in enumerate(folds[:NF]):
    if f<done: continue
    Xa,Xb,Tt=X.iloc[a].reset_index(drop=True),X.iloc[b].reset_index(drop=True),T.copy()
    if 'TE' in flags:
        ta,(tb,tt)=nested_te(PROD,y,a,[b,np.arange(n,len(df))],10); Xa['TE_PRODUCT']=ta; Xb['TE_PRODUCT']=tb; Tt['TE_PRODUCT']=tt
    pb=np.zeros(len(b)); ptf=np.zeros(len(T))
    for s in seeds:
        if which=='lgb':
            import lightgbm as lgb
            P=dict(objective='binary',learning_rate=0.01,num_leaves=16,max_depth=6,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l1=2,lambda_l2=10,min_gain_to_split=0.01,cat_smooth=50,cat_l2=20,max_cat_to_onehot=4,verbose=-1,n_jobs=4,seed=s)
            m=lgb.train(P,lgb.Dataset(Xa,y[a]),10000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
            p=m.predict(Xb,num_iteration=m.best_iteration); q=m.predict(Tt,num_iteration=m.best_iteration); it=m.best_iteration
        elif which=='cat':
            from catboost import CatBoostClassifier
            lr=float(os.environ.get('CAT_LR','0.03'))
            m=CatBoostClassifier(iterations=5000,learning_rate=lr,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,thread_count=4,verbose=0,random_seed=s)
            m.fit(Xa,y[a],cat_features=cats,eval_set=(Xb,y[b]),use_best_model=True)
            p=m.predict_proba(Xb)[:,1]; q=m.predict_proba(Tt)[:,1]; it=m.best_iteration_
        else:
            import xgboost as xgb
            m=xgb.XGBClassifier(n_estimators=10000,learning_rate=0.01,max_depth=5,min_child_weight=30,subsample=0.8,colsample_bytree=0.3,reg_lambda=10,reg_alpha=2,tree_method='hist',enable_categorical=True,max_cat_to_onehot=4,eval_metric='auc',early_stopping_rounds=300,n_jobs=4,random_state=s)
            m.fit(Xa,y[a],eval_set=[(Xb,y[b])],verbose=False)
            p=m.predict_proba(Xb)[:,1]; q=m.predict_proba(Tt)[:,1]; it=m.best_iteration
        pb+=p/len(seeds); ptf+=q/len(seeds)
        print(tag,'fold',f,'seed',s,round(roc_auc_score(y[b],p),5),it,f'{time.time()-t0:.0f}s',flush=True)
    oof[b]=pb; pt+=ptf/NF
    print(tag,'fold',f,'AUC',round(roc_auc_score(y[b],pb),5),flush=True)
    np.savez(CK,oof=oof,pt=pt,done=f+1)
fl=[round(roc_auc_score(y[b],oof[b]),5) for a,b in folds[:NF]]
if NF==5: print(tag,'OOF AUC',round(roc_auc_score(y,oof),5),'folds',fl,flush=True); np.save(f'{S}/oof_{tag}.npy',oof); np.save(f'{S}/pt_{tag}.npy',pt)
else: print(tag,'partial folds',fl,'mean',round(np.mean(fl),5),flush=True)

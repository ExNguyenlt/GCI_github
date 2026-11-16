import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,sys.argv[2]); from fe import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
df,y,n,cats,ids=load(); which=sys.argv[1]; out=sys.argv[2]
X=df.iloc[:n].reset_index(drop=True); T=df.iloc[n:].reset_index(drop=True)
oof=np.zeros(n); pt=np.zeros(len(T)); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
if which.startswith('lgb'):
    import lightgbm as lgb
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
    P=dict(objective='binary',learning_rate=0.01,num_leaves=16,max_depth=6,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l1=2,lambda_l2=10,min_gain_to_split=0.01,cat_smooth=50,cat_l2=20,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
    if which=='lgb_dart': P.update(boosting='gbdt',num_leaves=32,max_depth=-1,learning_rate=0.01,feature_fraction=0.2,extra_trees=True)
    for f,(a,b) in enumerate(folds):
        m=lgb.train(P,lgb.Dataset(X.iloc[a],y[a]),10000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[b]=m.predict(X.iloc[b],num_iteration=m.best_iteration); pt+=m.predict(T,num_iteration=m.best_iteration)/5
        print(which,f,round(roc_auc_score(y[b],oof[b]),4),m.best_iteration,flush=True)
elif which=='cat':
    from catboost import CatBoostClassifier
    for f,(a,b) in enumerate(folds):
        m=CatBoostClassifier(iterations=5000,learning_rate=0.03,depth=6,l2_leaf_reg=5,eval_metric='AUC',od_type='Iter',od_wait=300,thread_count=4,verbose=0,random_seed=0)
        m.fit(X.iloc[a],y[a],cat_features=cats,eval_set=(X.iloc[b],y[b]),use_best_model=True)
        oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(T)[:,1]/5
        print(which,f,round(roc_auc_score(y[b],oof[b]),4),m.best_iteration_,flush=True)
elif which=='xgb':
    import xgboost as xgb
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
    for f,(a,b) in enumerate(folds):
        m=xgb.XGBClassifier(n_estimators=10000,learning_rate=0.01,max_depth=5,min_child_weight=30,subsample=0.8,colsample_bytree=0.3,reg_lambda=10,reg_alpha=2,tree_method='hist',enable_categorical=True,max_cat_to_onehot=4,eval_metric='auc',early_stopping_rounds=300,n_jobs=4)
        m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],verbose=False)
        oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(T)[:,1]/5
        print(which,f,round(roc_auc_score(y[b],oof[b]),4),m.best_iteration,flush=True)
print(which,'OOF AUC',round(roc_auc_score(y,oof),5),flush=True)
np.save(f'{out}/oof_{which}.npy',oof); np.save(f'{out}/pt_{which}.npy',pt); np.save(f'{out}/y.npy',y); np.save(f'{out}/ids.npy',ids[n:])

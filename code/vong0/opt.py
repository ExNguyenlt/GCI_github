# Optuna tuning (fast lr) -> retrain best params at low lr with same 5 folds -> save oof/pt as *_opt
import sys, json, numpy as np, pandas as pd, warnings, optuna; warnings.filterwarnings('ignore')
S=sys.argv[2]; sys.path.insert(0,S); from fe import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
which=sys.argv[1]; NT=int(sys.argv[3])
df,y,n,cats,ids=load(); X=df.iloc[:n].reset_index(drop=True); T=df.iloc[n:].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
if which in('lgb','xgb'):
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
def fit(p,lr,fl,final=False):
    oof=np.zeros(n); pt=np.zeros(len(T))
    for a,b in fl:
        if which=='lgb':
            import lightgbm as lgb
            P=dict(objective='binary',learning_rate=lr,verbose=-1,n_jobs=4,bagging_freq=1,max_cat_to_onehot=4,**p)
            m=lgb.train(P,lgb.Dataset(X.iloc[a],y[a]),20000,valid_sets=[lgb.Dataset(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(int(3/lr),verbose=False)])
            oof[b]=m.predict(X.iloc[b],num_iteration=m.best_iteration)
            if final: pt+=m.predict(T,num_iteration=m.best_iteration)/5
        elif which=='xgb':
            import xgboost as xgb
            m=xgb.XGBClassifier(n_estimators=20000,learning_rate=lr,tree_method='hist',enable_categorical=True,max_cat_to_onehot=4,eval_metric='auc',early_stopping_rounds=int(3/lr),n_jobs=4,**p)
            m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],verbose=False)
            oof[b]=m.predict_proba(X.iloc[b])[:,1]
            if final: pt+=m.predict_proba(T)[:,1]/5
        else:
            from catboost import CatBoostClassifier
            m=CatBoostClassifier(iterations=20000,learning_rate=lr,eval_metric='AUC',od_type='Iter',od_wait=int(3/lr),thread_count=4,verbose=0,random_seed=0,**p)
            m.fit(X.iloc[a],y[a],cat_features=cats,eval_set=(X.iloc[b],y[b]),use_best_model=True)
            oof[b]=m.predict_proba(X.iloc[b])[:,1]
            if final: pt+=m.predict_proba(T)[:,1]/5
    idx=np.concatenate([b for _,b in fl])
    return roc_auc_score(y[idx],oof[idx]),oof,pt
def space(t):
    if which=='lgb': return dict(num_leaves=t.suggest_int('num_leaves',8,64,log=True),max_depth=t.suggest_int('max_depth',3,10),
        min_child_samples=t.suggest_int('min_child_samples',20,400,log=True),feature_fraction=t.suggest_float('feature_fraction',0.1,0.7),
        bagging_fraction=t.suggest_float('bagging_fraction',0.5,1.0),lambda_l1=t.suggest_float('lambda_l1',1e-3,20,log=True),
        lambda_l2=t.suggest_float('lambda_l2',1e-3,50,log=True),cat_smooth=t.suggest_float('cat_smooth',1,100,log=True),
        min_gain_to_split=t.suggest_float('min_gain_to_split',0,0.5))
    if which=='xgb': return dict(max_depth=t.suggest_int('max_depth',3,8),min_child_weight=t.suggest_float('min_child_weight',1,200,log=True),
        subsample=t.suggest_float('subsample',0.5,1.0),colsample_bytree=t.suggest_float('colsample_bytree',0.1,0.7),
        reg_lambda=t.suggest_float('reg_lambda',1e-2,50,log=True),reg_alpha=t.suggest_float('reg_alpha',1e-3,20,log=True),gamma=t.suggest_float('gamma',0,2))
    return dict(depth=t.suggest_int('depth',4,8),l2_leaf_reg=t.suggest_float('l2_leaf_reg',1,50,log=True),
        random_strength=t.suggest_float('random_strength',0,2),rsm=t.suggest_float('rsm',0.2,1.0),
        bagging_temperature=t.suggest_float('bagging_temperature',0,5),border_count=t.suggest_int('border_count',32,255))
fast=folds[:3]; LRF={'lgb':0.05,'xgb':0.05,'cat':0.08}[which]
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=0))
st.optimize(lambda t: fit(space(t),LRF,fast)[0],n_trials=NT,callbacks=[lambda s,t: print(which,'trial',t.number,round(t.value,5),'best',round(s.best_value,5),flush=True)])
print(which,'BEST',st.best_value,json.dumps(st.best_params),flush=True)
LR={'lgb':0.01,'xgb':0.01,'cat':0.03}[which]
auc,oof,pt=fit(st.best_params,LR,folds,final=True)
print(which,'OPT OOF AUC',round(auc,5),flush=True)
np.save(f'{S}/oof_{which}_opt.npy',oof); np.save(f'{S}/pt_{which}_opt.npy',pt)

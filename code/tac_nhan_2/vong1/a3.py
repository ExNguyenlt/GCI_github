import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad/'
sys.path.insert(0,S); from fe import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids=load()
for c in cats: df[c]=df[c].astype('category')
print('n features',df.shape[1])
# adversarial
adv=np.r_[np.zeros(n),np.ones(len(df)-n)]
fa=list(StratifiedKFold(3,shuffle=True,random_state=0).split(df,adv)); oa=np.zeros(len(df))
P=dict(objective='binary',learning_rate=0.1,num_leaves=31,feature_fraction=0.5,verbose=-1,n_jobs=4)
imp=0
for a,b in fa:
    m=lgb.train(P,lgb.Dataset(df.iloc[a],adv[a]),300); oa[b]=m.predict(df.iloc[b]); imp=imp+pd.Series(m.feature_importance('gain'),df.columns)
print('ADV AUC',round(A(adv,oa),4)); print((imp/imp.sum()).sort_values(ascending=False).head(6).round(3).to_dict())
# fold-0 quick LGB importance + ablations
X=df.iloc[:n].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)); a,b=folds[0]
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4,seed=0)
def run(cols,tag,seed=0):
    t=time.time(); Q=dict(P,seed=seed)
    m=lgb.train(Q,lgb.Dataset(X.loc[a,cols],y[a]),5000,valid_sets=[lgb.Dataset(X.loc[b,cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    p=m.predict(X.loc[b,cols],num_iteration=m.best_iteration); s=A(y[b],p); print(f'{tag:28s} AUC {s:.5f} it {m.best_iteration} {time.time()-t:.0f}s',flush=True); return m,s
allc=list(X.columns)
m,_=run(allc,'ALL seed0')
gi=pd.Series(m.feature_importance('gain'),allc); gi=gi/gi.sum()
print('top20 gain', gi.sort_values(ascending=False).head(20).round(3).to_dict())
print('n feat gain<0.3%:',(gi<0.003).sum(), 'zero:',(gi==0).sum())
for s in (1,2): run(allc,f'ALL seed{s}',s)
raw=[c for c in allc if c in pd.read_csv('/home/user/GCI_github/com/test.csv',nrows=1).columns]+['DAYS_EMPLOYED_ANOM']
run(raw,'RAW only (33 cols)')
run(raw+['CREDIT_ANNUITY','EXT_MEAN','EXT_PROD','EXT_W','EXT_NA','ANNUITY_INCOME','CREDIT_GOODS','DAYS_EMPLOYED_R_BIRTH'],'RAW+8 core FE')
top=gi.sort_values(ascending=False).index[:40].tolist(); run(top,'TOP40 by gain')

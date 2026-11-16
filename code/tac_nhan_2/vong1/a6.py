import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad/'
sys.path.insert(0,S); from fe import load
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids=load()
for c in cats: df[c]=df[c].astype('category')
# EXT imputation (no TARGET used), out-of-fold over train+test
EXTD=[c for c in df.columns if 'EXT' in c or c.startswith('E1') or c.startswith('E2')]
base=[c for c in df.columns if c not in EXTD]
for tgt in ['EXT_SOURCE_1','EXT_SOURCE_3','EXT_SOURCE_2']:
    others=[e for e in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if e!=tgt]
    cols=base+others; have=df[tgt].notna().values; pred=np.zeros(len(df))
    Pr=dict(objective='regression',learning_rate=0.1,num_leaves=31,feature_fraction=0.5,min_child_samples=50,verbose=-1,n_jobs=4)
    hv=np.where(have)[0]; nh=np.where(~have)[0]; cnt=np.zeros(len(df))
    for a,b in KFold(3,shuffle=True,random_state=0).split(hv):
        m=lgb.train(Pr,lgb.Dataset(df.iloc[hv[a]][cols],df[tgt].values[hv[a]]),400)
        pred[hv[b]]=m.predict(df.iloc[hv[b]][cols]); pred[nh]+=m.predict(df.iloc[nh][cols])/3
    r=np.corrcoef(pred[hv],df[tgt].values[hv])[0,1]; print(tgt,'imputer corr',round(r,3),'R2~',round(r*r,3),flush=True)
    df[tgt+'_PRED']=pred; df[tgt+'_FILL']=df[tgt].fillna(pd.Series(pred)); df[tgt+'_RES']=df[tgt]-pred
F=['EXT_SOURCE_1_FILL','EXT_SOURCE_2_FILL','EXT_SOURCE_3_FILL']
df['EXTF_MEAN']=df[F].mean(1); df['EXTF_PROD']=df[F].prod(1); df['EXTF_W']=df.EXT_SOURCE_1_FILL*2+df.EXT_SOURCE_2_FILL+df.EXT_SOURCE_3_FILL*3
df['TERM']=np.r_[np.load('term.npy'),np.full(len(df)-n,np.nan)]  # train only placeholder; recompute for all below
k=(df.AMT_CREDIT/df.AMT_ANNUITY).values
def rate(k,nn):
    lo=np.full_like(k,1e-6); hi=np.full_like(k,0.2)
    for _ in range(60):
        mid=(lo+hi)/2; f=(1-(1+mid)**-nn)/mid; lo=np.where(f>k,mid,lo); hi=np.where(f>k,hi,mid)
    r=(lo+hi)/2; r[k>=nn]=np.nan; return r
for nn in [12,24,36,48,60]: df[f'RATE_{nn}']=rate(k,nn)*12
X=df.iloc[:n].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
orig=[c for c in load.__globals__ and X.columns if not (c.endswith('_PRED') or c.endswith('_FILL') or c.endswith('_RES') or c.startswith('EXTF') or c.startswith('RATE') or c=='TERM')]
def run(cols,tag,fl=(0,1)):
    res=[]
    for f in fl:
        a,b=folds[f]
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(X.loc[a,cols],y[a]),5000,valid_sets=[lgb.Dataset(X.loc[b,cols],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(X.loc[b,cols],num_iteration=m.best_iteration)))
    print(f'{tag:30s}',np.round(res,5),'mean',round(np.mean(res),5),flush=True); return np.array(res)
imp=[c for c in X.columns if c.endswith(('_PRED','_FILL','_RES')) or c.startswith('EXTF')]; rt=[c for c in X.columns if c.startswith('RATE')]
b3=run(orig+imp+rt,'+IMPUTE+RATE')
print('vs BASE mean 0.75919, impute 0.76144, rate 0.76136')
m=lgb.train(dict(P,seed=0),lgb.Dataset(X.loc[folds[0][0],orig+imp+rt],y[folds[0][0]]),700)
g=pd.Series(m.feature_importance('gain'),orig+imp+rt); g=g/g.sum(); print('gain share new:',g[imp+rt].sort_values(ascending=False).round(3).to_dict())

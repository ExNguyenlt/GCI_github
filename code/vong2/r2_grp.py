# L2.4b: GBDT rieng cho nhom yeu o tang 1. Voi moi nhom g: LGB (tham so nhu run_r2, lr 0.02) hoc CHI tren dong thuoc g cua phan train fold, du doan dong thuoc g cua fold valid + test.
# Luu oof_sub_<g>_r2.npy (NaN ngoai nhom), pt_sub_<g>_r2.npy. python3 -I r2_grp.py <spec>
import sys, os, re, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
newc=set(sum(G.values(),[])); cols=[c for c in df.columns if c not in newc]+G['IMP']+G['RATE']+G['TERM']
for sgn,nm in re.findall(r'([+-])([A-Za-z0-9_]+)',sys.argv[1]):
    cc=G[nm] if nm in G else [nm]
    cols=cols+[c for c in cc if c not in cols] if sgn=='+' else [c for c in cols if c not in cc]
for c in cats: df[c]=df[c].astype('category')
age=-df.DAYS_BIRTH.values/365.25
GR={'old':(df.DAYS_EMPLOYED_ANOM.values==1)|(age>60),'rev':(df.NAME_CONTRACT_TYPE!='Cash loans').values,'young':age<30,'ext2na':df.EXT_NA.values>=2}
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.02,num_leaves=16,max_depth=6,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l1=2,lambda_l2=10,min_gain_to_split=0.01,cat_smooth=50,cat_l2=20,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
X=df[cols]; ref=np.load(S+'/oof_lgb_r2.npy') if os.path.exists(S+'/oof_lgb_r2.npy') else None
for g,mask in GR.items():
    mtr=mask[:n]; mte=mask[n:]; oof=np.full(n,np.nan); pt=np.zeros(mte.sum()); t0=time.time()
    for a,b in folds:
        ga=a[mtr[a]]; gb=b[mtr[b]]; pb=np.zeros(len(gb))
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(X.iloc[ga],y[ga]),6000,valid_sets=[lgb.Dataset(X.iloc[gb],y[gb])],callbacks=[lgb.early_stopping(200,verbose=False)])
            pb+=m.predict(X.iloc[gb],num_iteration=m.best_iteration)/2; pt+=m.predict(X.iloc[n:][mte],num_iteration=m.best_iteration)/10
        oof[gb]=pb
    full=np.full(len(df)-n,np.nan); full[mte]=pt; np.save(f'{S}/oof_sub_{g}_r2.npy',oof); np.save(f'{S}/pt_sub_{g}_r2.npy',full)
    s=f'group {g}: n {mtr.sum()} ({mtr.mean():.3f}) dr {y[mtr].mean():.3f} AUC sub-model in-group {A(y[mtr],oof[mtr]):.5f}'
    if ref is not None: s+=f' | lgb_r2 in-group {A(y[mtr],ref[mtr]):.5f} | per-fold d {np.round([A(y[b[mtr[b]]],oof[b[mtr[b]]])-A(y[b[mtr[b]]],ref[b[mtr[b]]]) for a,b in folds],4)}'
    print(s,f'{time.time()-t0:.0f}s',flush=True)

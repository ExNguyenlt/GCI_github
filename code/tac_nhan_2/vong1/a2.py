import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
from scipy.stats import rankdata
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad/'
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
te=pd.read_csv(D+'test.csv')
for d in (tr,te): d.columns=[c.replace('﻿','') for c in d.columns]
y=np.load(S+'y.npy'); assert (tr.TARGET.values==y).all()
M=['lgb','cat','xgb','lgb_opt','cat_opt','xgb_opt']
o=np.mean([rankdata(np.load(S+f'oof_{m}.npy')) for m in M],0)
print('n',len(tr),len(te),'rate',round(y.mean(),4))
def grp(name,g):
    print('==',name)
    for k in pd.Series(g).dropna().unique():
        ix=(g==k)
        if ix.sum()<500: continue
        print(f'  {str(k)[:30]:30s} n={ix.sum():6d} share={ix.mean():.3f} rate={y[ix].mean():.4f} AUC={A(y[ix],o[ix]):.4f}')
print('overall',round(A(y,o),5))
grp('EXT1 missing',tr.EXT_SOURCE_1.isna().values)
grp('EXT3 missing',tr.EXT_SOURCE_3.isna().values)
grp('EXT NA count',tr[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].isna().sum(1).values)
grp('contract',tr.NAME_CONTRACT_TYPE.values)
grp('gender',tr.CODE_GENDER.values)
grp('emp anomaly',(tr.DAYS_EMPLOYED==365243).values)
grp('age bins',pd.cut(-tr.DAYS_BIRTH/365,[0,30,40,50,60,100]).astype(str).values)
# missing rate train vs test
mi=pd.DataFrame({'tr':tr.drop(columns='TARGET').isna().mean(),'te':te.isna().mean()}); print(mi[(mi.tr>0)|(mi.te>0)].round(3))
# single feature AUCs
print('single-feature AUC (|AUC-0.5|)')
r={}
for c in tr.columns:
    if c in('TARGET',) or tr[c].dtype==object or str(tr[c].dtype)=='str': continue
    v=tr[c].fillna(tr[c].median()).values
    try: r[c]=abs(A(y,v)-0.5)+0.5
    except: pass
print(pd.Series(r).sort_values(ascending=False).round(4).head(15))
# contribution check: EXT_SOURCE only logistic
from sklearn.linear_model import LogisticRegression
E=tr[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
print('EXT_MEAN AUC',round(1-A(y,E.mean(1).fillna(0.5)),4))
# duplicates
F=[c for c in tr.columns if c not in('SK_ID_CURR','TARGET')]
print('dup rows within train (features)',tr.duplicated(F).sum(),' train-test overlap',pd.merge(tr[F].drop_duplicates(),te[F].drop_duplicates(),on=F).shape[0])
# IDs distribution
print('id ranges', tr.SK_ID_CURR.min(),tr.SK_ID_CURR.max(),te.SK_ID_CURR.min(),te.SK_ID_CURR.max())
# CREDIT/ANNUITY term structure
t=tr.AMT_CREDIT/tr.AMT_ANNUITY; print('term rounded value counts top', (t.round()).value_counts().head(8).to_dict())
print('AUC credit/annuity', round(A(y,t.fillna(t.median())),4))

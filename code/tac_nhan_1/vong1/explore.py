import pandas as pd, numpy as np
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
tr.columns=[c.replace('﻿','') for c in tr.columns]
C=tr.AMT_CREDIT.values; A=tr.AMT_ANNUITY.values; y=tr.TARGET.values
ratio=C/A
print(pd.Series(ratio).describe())
print(tr.NAME_CONTRACT_TYPE.value_counts())
# solve monthly rate for each term by bisection
def rate(n):
    lo=np.zeros_like(C); hi=np.full_like(C,0.2)
    for _ in range(60):
        mid=(lo+hi)/2
        pv=A*(1-(1+mid)**-n)/np.maximum(mid,1e-12)
        lo=np.where(pv>C,mid,lo); hi=np.where(pv>C,hi,mid)
    r=(lo+hi)/2
    r[A*n<C]=np.nan
    return r
from sklearn.metrics import roc_auc_score
cash=(tr.NAME_CONTRACT_TYPE=='Cash loans').values
for n in [6,10,12,18,24,30,36,42,48,54,60]:
    r=rate(n)
    v=r[cash]; ok=~np.isnan(v)
    # how many rows give annual rate in plausible range
    print(n, 'valid',ok.mean().round(3), 'annual quantiles', np.nanquantile(v*12,[.05,.25,.5,.75,.95]).round(3))
# look for exactness: annuity*n - credit modulo patterns
print((tr.AMT_ANNUITY*2 % 1==0).mean(), (tr.AMT_CREDIT%1==0).mean())
print(tr.AMT_ANNUITY.head(20).tolist())
print(tr.AMT_GOODS_PRICE.isna().mean(), ((tr.AMT_CREDIT-tr.AMT_GOODS_PRICE)/tr.AMT_GOODS_PRICE).describe())

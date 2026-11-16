import pandas as pd, numpy as np
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
tr.columns=[c.replace('﻿','') for c in tr.columns]
C=tr.AMT_CREDIT.values; A=tr.AMT_ANNUITY.values; y=tr.TARGET.values
def rate(n):
    lo=np.zeros_like(C); hi=np.full_like(C,0.2)
    for _ in range(60):
        mid=(lo+hi)/2
        pv=A*(1-(1+mid)**-n)/np.maximum(mid,1e-12)
        lo=np.where(pv>C,mid,lo); hi=np.where(pv>C,hi,mid)
    r=(lo+hi)/2; r[A*n<C]=np.nan; return r
cash=(tr.NAME_CONTRACT_TYPE=='Cash loans').values
R={n:rate(n) for n in [12,18,24,30,36,48,60]}
for n,r in R.items():
    a=np.round(r[cash]*12,3)
    vc=pd.Series(a).value_counts()
    print(n,'top annual rate values (share):', [(k, round(v/cash.sum(),4)) for k,v in vc.head(6).items()])
# revolving
rv=~cash
print('revolving ratio C/A', pd.Series(np.round(C[rv]/A[rv],3)).value_counts().head(8))
# default rate by TERM_ROUND bins for cash
s=pd.DataFrame({'t':np.round(C/A,1),'y':y})[cash]
vc=s.t.value_counts(); print(vc.head(15))

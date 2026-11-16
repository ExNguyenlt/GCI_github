import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as A
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
tr.columns=[c.replace('﻿','') for c in tr.columns]; y=tr.TARGET.values
k=(tr.AMT_CREDIT/tr.AMT_ANNUITY).values
def rate(k,nn):
    # solve (1-(1+r)^-n)/r = k for r by bisection; r in (0,0.2)
    lo=np.full_like(k,1e-6); hi=np.full_like(k,0.2)
    for _ in range(60):
        mid=(lo+hi)/2; f=(1-(1+mid)**-nn)/mid
        lo=np.where(f>k,mid,lo); hi=np.where(f>k,hi,mid)
    r=(lo+hi)/2; r[k>=nn]=np.nan; return r
for ct in ['Cash loans','Revolving loans']:
    ix=(tr.NAME_CONTRACT_TYPE==ct).values
    print(ct,'k quantiles',np.nanpercentile(k[ix],[1,10,50,90,99]).round(2), 'AUC(k)',round(A(y[ix],np.nan_to_num(k[ix],nan=20)),4))
ix=(tr.NAME_CONTRACT_TYPE=='Cash loans').values
for nn in [12,18,24,30,36,48,60]:
    r=rate(k,nn)*12; v=ix&~np.isnan(r)
    h=np.histogram(r[v],bins=[0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,1,3])[0]
    print(nn,'valid',v.sum(),'annual r hist',h)
# heuristic: pick smallest standard term n in {6,10,12,18,24,30,36,42,48,54,60} for which implied annual rate is in [0.0,0.6]
Ns=np.array([6,10,12,18,24,30,36,42,48,54,60])
R=np.stack([rate(k,nn)*12 for nn in Ns],1)
ok=(R>0.08)&(R<0.6)
first=np.where(ok.any(1),ok.argmax(1),-1)
term=np.where(first>=0,Ns[np.clip(first,0,None)],np.nan); rr=np.where(first>=0,R[np.arange(len(k)),np.clip(first,0,None)],np.nan)
print('term counts',pd.Series(term).value_counts(dropna=False).head(12).to_dict())
for nm,v in [('term',term),('rate',rr)]:
    m=~np.isnan(v); print(nm,'AUC',round(A(y[m],v[m]),4),'coverage',m.mean().round(3))
# within term group, does rate matter?
for t in [12,24,36,48,60]:
    m=(term==t)
    if m.sum()>2000: print(' term',t,'n',m.sum(),'rate AUC',round(A(y[m],rr[m]),4),'default rate',y[m].mean().round(4))
np.save('term.npy',term); np.save('rate.npy',rr)

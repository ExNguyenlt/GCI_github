import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
te=pd.read_csv(D+'test.csv')
for d in (tr,te): d.columns=[c.replace('﻿','') for c in d.columns]
y=tr.pop('TARGET').values; n0=len(tr); df=pd.concat([tr,te],ignore_index=True)
C=df.AMT_CREDIT.values; A=df.AMT_ANNUITY.values
def rate(n):
    lo=np.zeros_like(C); hi=np.full_like(C,0.2)
    for _ in range(60):
        mid=(lo+hi)/2
        pv=A*(1-(1+mid)**-n)/np.maximum(mid,1e-12)
        lo=np.where(pv>C,mid,lo); hi=np.where(pv>C,hi,mid)
    r=(lo+hi)/2; r[A*n<C]=np.nan; return r
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
terms=[6,10,12,18,24,30,36,42,48,54,60]
dens=[];rates=[]
for n in terms:
    r=rate(n); key=np.round(r*12*400)  # 0.25% annual bins
    s=pd.Series(key); cnt=s.map(s.value_counts()).fillna(0).values
    dens.append(cnt); rates.append(r*12)
dens=np.vstack(dens).T; rates=np.vstack(rates).T
j=dens.argmax(1); term=np.array(terms)[j]; rt=rates[np.arange(len(j)),j]; dmax=dens.max(1)
term[~cash]=0
ca=C/A
ytr=y; m=cash[:n0]
def auc(x):
    x=x[:n0][m]; ok=~np.isnan(x); a=roc_auc_score(ytr[m][ok],x[ok]); return round(max(a,1-a),4)
print('C/A',auc(ca),'term_hat',auc(term.astype(float)),'rate_hat',auc(rt),'dens',auc(dmax.astype(float)))
print(pd.Series(term[:n0][m]).value_counts())
print(pd.DataFrame({'t':term[:n0][m],'y':y[m]}).groupby('t').y.mean())
# residual info: within C/A bins, does rate_hat add?
d=pd.DataFrame({'ca':ca[:n0][m],'rt':rt[:n0][m],'y':y[m],'t':term[:n0][m]})
d['cab']=pd.qcut(d.ca,20,labels=False,duplicates='drop')
for b,g in list(d.groupby('cab'))[::4]:
    ok=g.rt.notna()
    if g.y[ok].nunique()>1: print('bin',b,'n',len(g),'AUC rate within bin',round(roc_auc_score(g.y[ok],g.rt[ok]),4),'AUC term within',round(roc_auc_score(g.y,g.t),4))
# exact-value frequency of C/A
k=pd.Series(np.round(ca,3)); f=k.map(k.value_counts()).values.astype(float)
print('freq C/A exact',auc(f))

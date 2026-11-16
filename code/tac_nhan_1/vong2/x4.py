import sys,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score as AUC
S=sys.argv[1]
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
tr=df.iloc[:n]
def au(x,m=None):
    x=np.asarray(x,float); ok=~np.isnan(x) if m is None else (~np.isnan(x))&m
    a=AUC(y[ok],x[ok]); return round(a,4)
for c in ['AMT_INCOME_TOTAL','AMT_GOODS_PRICE','AMT_CREDIT','AMT_ANNUITY']:
    x=tr[c].values
    for m in [1,10,100,450,1000,4500,9000,22500,45000,225000]:
        f=(np.mod(x,m)==0)
        if f.mean()>0.002 and f.mean()<0.998: print(c,m,'share',round(f.mean(),3),'dr',round(y[f].mean(),4),'vs',round(y[~f].mean(),4))
    # decimals
    print(c,'non-integer share',round(np.mean(x%1!=0),3), 'dr', round(y[x%1!=0].mean(),4) if (x%1!=0).any() else None)
# top values income
print(pd.Series(tr.AMT_INCOME_TOTAL).value_counts().head(12).to_dict())
print(pd.Series(tr.AMT_GOODS_PRICE).value_counts().head(12).to_dict())

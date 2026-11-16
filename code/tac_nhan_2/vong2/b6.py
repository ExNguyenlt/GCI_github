import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1(); X=df.iloc[:n]
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
k=(df.AMT_CREDIT/df.AMT_ANNUITY)[cash]
print('k quantiles',np.round(np.quantile(k.dropna(),[0,.01,.1,.25,.5,.75,.9,.99,1]),2))
print('TERM_HAT NaN among cash',round(df.TERM_HAT[cash].isna().mean(),4))
print('RATE_HAT quantiles',np.round(np.nanquantile(df.RATE_HAT[cash],[.01,.1,.25,.5,.75,.9,.99]),3))
print('share RATE_HAT>0.35',round((df.RATE_HAT[cash]>0.35).mean(),3))
s1=np.load(S+'/a2r2/s1.npy')
# calib theo decile
q=pd.qcut(s1,10,labels=False); print('default rate by decile of s1:',np.round(pd.Series(y).groupby(q).mean().values,3))
print('top decile captures',round(y[q==9].sum()/y.sum(),3),' top 30%',round(y[q>=7].sum()/y.sum(),3))
# seeds: lgb 3 seed vs xgb 1 seed -> per-fold gain of lgb over xgb

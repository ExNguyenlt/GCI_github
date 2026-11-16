import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.metrics import roc_auc_score as AUC
df,y,n,cats,ids,g=load_r1()
df.to_pickle(S+'/loop/a1r2/df_r1.pkl'); np.save(S+'/loop/a1r2/y.npy',y)
C=df.AMT_CREDIT.values; G=df.AMT_GOODS_PRICE.values; A=df.AMT_ANNUITY.values
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
cg=C/G
print('cash share',cash.mean())
for nm,m in [('cash',cash),('rev',~cash)]:
    s=pd.Series(np.round(cg[m],4)); print(nm,'C/G top', s.value_counts().head(15).to_dict())
    print(nm,'C==G',np.mean(np.isclose(C[m],G[m])), 'C<G',np.mean(C[m]<G[m]-1),'C>G',np.mean(C[m]>G[m]+1))

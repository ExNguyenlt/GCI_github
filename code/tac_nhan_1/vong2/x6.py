import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S)
from fe_r1 import rate
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl')
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values; ins=cash&(C>G+1)
for nm,P in [('C',C),('G',G)]:
  for nn in [12,18,24,36,48,60]:
    r=rate(P,A,nn)*12; r[~ins]=np.nan
    s=pd.Series(np.round(r*1e4)); vc=s.value_counts()
    print(nm,nn,'insured rows share in top50 fine bins',round(vc.head(50).sum()/ins.sum(),3),'top',vc.head(3).to_dict())

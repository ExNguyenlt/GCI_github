import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S)
from fe_r1 import rate
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values; A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
print('TERM_DENS_RATIO quantiles', np.nanquantile(df.TERM_DENS_RATIO.values,[.1,.25,.5,.75,.9]))
for nn in [12,24,36,48,60]:
    r=rate(C,A,nn)*12; r[~cash]=np.nan
    s=pd.Series(np.round(r*1e4)/1e4)   # 0.01% bins
    vc=s.value_counts(); print(nn,'top fine bins',vc.head(8).to_dict(),'share top20',round(vc.head(20).sum()/cash.sum(),3))

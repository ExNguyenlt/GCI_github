import sys,numpy as np,pandas as pd
S=sys.argv[1]
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
C=df.AMT_CREDIT.values; G=df.AMT_GOODS_PRICE.values; A=df.AMT_ANNUITY.values
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
k=(C/G-1)/0.0132; kr=np.round(k); err=np.abs(k-kr)
print('frac |k-round|<0.02 among C>G cash:', np.mean(err[cash&(C>G+1)]<0.02))
print(pd.Series(kr[cash&(C>G+1)&(err<0.02)]).value_counts().sort_index().to_dict())
th=df.TERM_HAT.values
m=cash&(C>G+1)&(err<0.02)
print(pd.crosstab(kr[m],th[m]).iloc[:40].to_string())

import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S+'/loop/a1r2')
from harness import base,gain,oe
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
b=base(S)
for e in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
    s=df[e]; print(e,'nunique',s.nunique(),'na',round(s.isna().mean(),3))
    cnt=s.map(s.value_counts()).values[:n]
    q=np.where(np.isnan(cnt),-1,np.minimum(cnt,50)); qb=pd.cut(q,[-2,0,1,2,5,10,20,1e9],labels=False)
    print(pd.Series(q).describe()[['mean','50%']].to_dict()); print(oe(y,b,qb).to_string()); print('gain',gain(y,b,pd.DataFrame({'q':qb})))
print('REGION nunique',df.REGION_POPULATION_RELATIVE.nunique())

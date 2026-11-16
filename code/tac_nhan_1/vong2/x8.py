import sys,numpy as np,pandas as pd
S=sys.argv[1]
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
d=df[['EXT_SOURCE_2','DAYS_BIRTH','DAYS_ID_PUBLISH','DAYS_REGISTRATION','CODE_GENDER','REGION_POPULATION_RELATIVE','AMT_INCOME_TOTAL','ORGANIZATION_TYPE']].copy()
d['k_id']=d.DAYS_BIRTH-d.DAYS_ID_PUBLISH
g=d.groupby('EXT_SOURCE_2')
d['cnt']=g.EXT_SOURCE_2.transform('size')
sub=d[d.cnt==2].sort_values('EXT_SOURCE_2')
a=sub.iloc[::2].reset_index(drop=True); b=sub.iloc[1::2].reset_index(drop=True)
print('pairs',len(a))
for c in ['DAYS_BIRTH','k_id','CODE_GENDER','REGION_POPULATION_RELATIVE','AMT_INCOME_TOTAL','ORGANIZATION_TYPE']:
    print(c,'same share',round(np.mean(a[c].values==b[c].values),4))
print(a.head(5).to_string()); print(b.head(5).to_string())

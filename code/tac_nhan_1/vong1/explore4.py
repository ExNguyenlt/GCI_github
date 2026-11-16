import pandas as pd, numpy as np
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
te=pd.read_csv(D+'test.csv')
for d in (tr,te): d.columns=[c.replace('﻿','') for c in d.columns]
y=tr.pop('TARGET'); n0=len(tr); df=pd.concat([tr,te],ignore_index=True); df['y']=list(y)+[np.nan]*len(te)
df['k1']=df.DAYS_BIRTH-df.DAYS_ID_PUBLISH
df['k2']=df.DAYS_BIRTH-df.DAYS_REGISTRATION
df['k3']=df.DAYS_BIRTH-df.DAYS_EMPLOYED.where(df.DAYS_EMPLOYED!=365243)
for keys in [['k1','CODE_GENDER'],['k1','k2','CODE_GENDER'],['k1','k3','CODE_GENDER'],['k1','k2','k3','CODE_GENDER']]:
    g=df.groupby(keys,dropna=False).size()
    sz=df.set_index(keys).index.map(g)
    print(keys,'rows in groups>1:',(np.array(sz)>1).mean().round(4))
keys=['k1','CODE_GENDER','EXT_SOURCE_1']
d=df[df.EXT_SOURCE_1.notna()]
g=d.groupby(keys).size(); print('k1+gender+EXT1 rows in dup groups',(d.set_index(keys).index.map(g)>1).mean())
keys=['k1','CODE_GENDER','NAME_EDUCATION_TYPE']
g=df.groupby(keys).size()
dup=df[np.array(df.set_index(keys).index.map(g))>1].sort_values(keys)
print(dup[keys+['DAYS_BIRTH','DAYS_EMPLOYED','AMT_INCOME_TOTAL','EXT_SOURCE_2','EXT_SOURCE_3','ORGANIZATION_TYPE','y']].head(20).to_string())
# Target rate in duplicated groups (k1,k2,gender)
keys=['k1','k2','CODE_GENDER']
df['gs']=np.array(df.set_index(keys).index.map(df.groupby(keys).size()))
print(df[:n0].groupby(np.minimum(df.gs[:n0],4)).y.agg(['mean','size']))

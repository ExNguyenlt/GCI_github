import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
tr.columns=[c.replace('﻿','') for c in tr.columns]
y=tr.TARGET.values
age=-tr.DAYS_BIRTH/365.25; idage=(tr.DAYS_ID_PUBLISH-tr.DAYS_BIRTH)/365.25
print(pd.cut(idage,[0,14,16,20,20.5,21,25,30,45,45.5,46,50,70]).value_counts().sort_index())
d=pd.DataFrame({'age':age,'idage':idage,'y':y})
d['since20']=d.idage-20; d['since45']=d.idage-45
# expected renewal: age>=45 but id issued before 45 -> overdue
d['overdue45']=((d.age>=45)&(d.idage<45)).astype(int)
d['overdue20']=((d.age>=20)&(d.idage<20)).astype(int)
d['late']=np.where(d.idage>=45,d.idage-45,np.where(d.idage>=20,d.idage-20,d.idage-14))
print(d.groupby('overdue45').y.agg(['mean','size']))
print(d.groupby(pd.cut(d.late,[-1,0.1,0.25,0.5,1,2,5,10,30])).y.agg(['mean','size']))
# within age bins effect of 'late'
d['ab']=pd.cut(d.age,[20,25,30,40,45,50,70])
for b,g in d.groupby('ab',observed=True):
    a=roc_auc_score(g.y,g.late); print(b,len(g),'AUC late',round(a,4), 'overdue45 rate',g.overdue45.mean().round(3))

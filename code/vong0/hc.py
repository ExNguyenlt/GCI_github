import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
D='/home/user/GCI_github/com/'
tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
te=pd.read_csv(D+'test.csv')
tr.columns=[c.replace('﻿','') for c in tr.columns]; te.columns=[c.replace('﻿','') for c in te.columns]
y=tr.pop('TARGET'); n=len(tr); df=pd.concat([tr,te],ignore_index=True)
df['DAYS_EMPLOYED_ANOM']=(df.DAYS_EMPLOYED==365243).astype(int); df.loc[df.DAYS_EMPLOYED==365243,'DAYS_EMPLOYED']=np.nan
e=df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
df['EXT_MEAN']=e.mean(1); df['EXT_STD']=e.std(1); df['EXT_PROD']=e.prod(1,min_count=3); df['EXT_MIN']=e.min(1); df['EXT_MAX']=e.max(1)
df['EXT_W']=df.EXT_SOURCE_1.fillna(df.EXT_MEAN)*2+df.EXT_SOURCE_2.fillna(df.EXT_MEAN)+df.EXT_SOURCE_3.fillna(df.EXT_MEAN)*3
df['EXT_NA']=e.isna().sum(1)
df['CREDIT_ANNUITY']=df.AMT_CREDIT/df.AMT_ANNUITY; df['CREDIT_GOODS']=df.AMT_CREDIT/df.AMT_GOODS_PRICE
df['GOODS_MINUS']=df.AMT_GOODS_PRICE-df.AMT_CREDIT
df['ANNUITY_INCOME']=df.AMT_ANNUITY/df.AMT_INCOME_TOTAL; df['CREDIT_INCOME']=df.AMT_CREDIT/df.AMT_INCOME_TOTAL
df['INC_PER_PERSON']=df.AMT_INCOME_TOTAL/df.CNT_FAM_MEMBERS
df['EMP_BIRTH']=df.DAYS_EMPLOYED/df.DAYS_BIRTH; df['CAR_BIRTH']=df.OWN_CAR_AGE/df.DAYS_BIRTH
df['PHONE_BIRTH']=df.DAYS_LAST_PHONE_CHANGE/df.DAYS_BIRTH; df['ID_BIRTH']=df.DAYS_ID_PUBLISH/df.DAYS_BIRTH
df['REG_BIRTH']=df.DAYS_REGISTRATION/df.DAYS_BIRTH
df['EXT_x_CA']=df.EXT_MEAN*df.CREDIT_ANNUITY; df['EXT2_BIRTH']=df.EXT_SOURCE_2*df.DAYS_BIRTH
cats=df.select_dtypes(exclude='number').columns.tolist()
for c in cats: df[c]=df[c].astype('category')
for c in ['ORGANIZATION_TYPE','OCCUPATION_TYPE']:
    g=df.groupby(c,observed=True)
    df[c+'_EXT_MEAN']=g.EXT_MEAN.transform('mean'); df[c+'_INC_REL']=df.AMT_INCOME_TOTAL/g.AMT_INCOME_TOTAL.transform('median')
feats=[c for c in df.columns if c!='SK_ID_CURR']
X=df.iloc[:n][feats]; T=df.iloc[n:][feats]
P=dict(objective='binary',learning_rate=0.02,num_leaves=24,min_child_samples=80,feature_fraction=0.5,bagging_fraction=0.8,bagging_freq=1,lambda_l1=1,lambda_l2=5,max_cat_to_onehot=4,cat_smooth=20,verbose=-1,n_jobs=-1)
oof=np.zeros(n); pt=np.zeros(len(T))
for seed in [0,1,2]:
  for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)):
    m=lgb.train({**P,'seed':seed},lgb.Dataset(X.iloc[a],y.iloc[a]),5000,valid_sets=[lgb.Dataset(X.iloc[b],y.iloc[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    p=m.predict(X.iloc[b],num_iteration=m.best_iteration); oof[b]+=p/3; pt+=m.predict(T,num_iteration=m.best_iteration)/15
    print(seed,f,round(roc_auc_score(y.iloc[b],p),4),m.best_iteration,flush=True)
print('OOF AUC',roc_auc_score(y,oof))
sub=pd.DataFrame({'SK_ID_CURR':te.SK_ID_CURR,'TARGET':pt})
sub.to_csv(sys.argv[1],index=False); print(sub.shape, sub.TARGET.describe())

# Dac trung target-free moi: (a) phan du "gia" (RATE_HAT, C/A, C/G, thu nhap, tin dung ...) so voi ho so khach; (b) mod 7 cua DAYS; (c) cum KMeans/GMM
from base import *
from sklearn.model_selection import KFold
raw=['CODE_GENDER','CNT_CHILDREN','AMT_INCOME_TOTAL','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','FLAG_EMP_PHONE','FLAG_WORK_PHONE','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','ORGANIZATION_TYPE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','NAME_CONTRACT_TYPE']
PR=dict(objective='regression',learning_rate=0.15,num_leaves=31,min_child_samples=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,n_jobs=2)
def oofreg(t,feats,rounds=150):
    ok=np.isfinite(t); pred=np.full(len(df),np.nan); hv=np.where(ok)[0]; nh=np.where(~ok)[0]; acc=np.zeros(len(nh))
    for a,b in KFold(3,shuffle=True,random_state=0).split(hv):
        m=lgb.train(PR,lgb.Dataset(df.iloc[hv[a]][feats],t[hv[a]]),rounds); pred[hv[b]]=m.predict(df.iloc[hv[b]][feats])
        if len(nh): acc+=m.predict(df.iloc[nh][feats])/3
    if len(nh): pred[nh]=acc
    r2=1-np.nanvar(t[ok]-pred[ok])/np.nanvar(t[ok]); return pred,r2
O={}
C=df.AMT_CREDIT.values.astype(float); An=df.AMT_ANNUITY.values.astype(float); Gd=df.AMT_GOODS_PRICE.values.astype(float)
cust=[c for c in raw if c!='NAME_CONTRACT_TYPE']
tg={'RATEHAT':df.RATE_HAT.values.astype(float),'LCA':np.log(C/An)}
for k,t in tg.items():
    p,r2=oofreg(t,cust+(['AMT_CREDIT'] if k in('RATEHAT','LCA','CG','TERMHAT') else [])); print(k,'R2 customer-only',round(r2,4),flush=True)
    O['PR_'+k+'_PRED']=p; O['PR_'+k+'_RES']=t-p
# thu nhap bat thuong so voi ho so (khong dung AMT)
noamt=[c for c in cust if c!='AMT_INCOME_TOTAL']
for k,t in {'LINC':np.log(df.AMT_INCOME_TOTAL.values.astype(float)),'EMP':np.where(df.DAYS_EMPLOYED.values>0,np.nan,df.DAYS_EMPLOYED.values).astype(float)}.items():
    f=[c for c in noamt if not (k=='EMP' and c in('DAYS_EMPLOYED','FLAG_EMP_PHONE'))]
    p,r2=oofreg(t,f); print(k,'R2',round(r2,4),flush=True); O['PR_'+k+'_PRED']=p; O['PR_'+k+'_RES']=t-p
for c in ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:
    v=df[c].values.astype(float); O['M7_'+c]=np.where(np.isfinite(v)&(v<=0),np.mod(-v,7),np.nan)
O['M7_ID_REG']=np.mod(df.DAYS_ID_PUBLISH.values-df.DAYS_REGISTRATION.values,7)
pd.to_pickle(pd.DataFrame(O),'f2.pkl'); print(pd.DataFrame(O).describe().T[['count','mean','std']])

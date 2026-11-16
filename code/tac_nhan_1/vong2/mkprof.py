# Tao cac bien the "ho so quan the cua san pham" (target-free, train+test). Dung: mkprof.py S
import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S)
from fe_r1 import rate
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl')
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
core=pd.DataFrame({'e':df.EXTF_MEAN.values,'e2':df.EXT_SOURCE_2.values,'e3':df.EXT_SOURCE_3.values,'e1p':df.EXT_SOURCE_1_PRED.values,'age':df.DAYS_BIRTH.values,
   'emp':df.DAYS_EMPLOYED.values,'male':(df.CODE_GENDER=='M').values.astype(float)})
wide=core.copy()
wide['inc']=np.log(df.AMT_INCOME_TOTAL.values); wide['cr']=np.log(C); wide['ins']=(C>G+1).astype(float); wide['cg']=C/G
for c in ['DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE','OWN_CAR_AGE','CNT_CHILDREN','REGION_RATING_CLIENT','REGION_POPULATION_RELATIVE','FLAG_WORK_PHONE','AMT_REQ_CREDIT_BUREAU_YEAR','EXT_SOURCE_3_PRED','EXT_SOURCE_2_PRED','ANNUITY_INCOME','CREDIT_INCOME','DAYS_EMPLOYED_ANOM']:
    wide[c]=df[c].values
wide['e3na']=df.EXT_SOURCE_3.isna().values.astype(float); wide['car_na']=df.OWN_CAR_AGE.isna().values.astype(float)
for c,vals in [('NAME_EDUCATION_TYPE',['Higher education','Secondary / secondary special']),('NAME_INCOME_TYPE',['Working','Pensioner','Commercial associate','State servant']),
               ('NAME_FAMILY_STATUS',['Married','Single / not married']),('NAME_HOUSING_TYPE',['House / apartment','With parents']),('OCCUPATION_TYPE',['Laborers','Drivers','Managers','Core staff','NA'])]:
    for v in vals: wide[c[:8]+'_'+v[:6]]=(df[c].astype(str)==v).values.astype(float)
def profile(p,cols,pre,minc=30,std=True):
    p=pd.Series(p).fillna(-9).values; d=cols.copy(); d['p']=p; s=pd.Series(p); cnt=s.map(s.value_counts()).values
    g=d.groupby('p'); o=pd.DataFrame({f'{pre}_{c}':g[c].transform('mean').values for c in cols.columns})
    if std: o[pre+'_estd']=g.e.transform('std').values
    o[cnt<minc]=np.nan; return o
th=df.TERM_HAT.values; rh=df.RATE_HAT.values
kTR=lambda m: np.where(cash,th*100000+np.round(rh*m),-5)
out={}
out['TRwide']=profile(kTR(200),wide,'PW')
out['TRmulti']=pd.concat([profile(kTR(m),core,f'PT{m}') for m in (50,200,1000)],axis=1)
rn=[]
for nn in (12,24,36,48,60):
    r=rate(C,A,nn)*12; rn.append(profile(np.where(cash&~np.isnan(r),np.round(r*200),-5),core[['e','e3','age','emp']],f'PR{nn}',std=False))
out['RATEn']=pd.concat(rn,axis=1)
out['TR_CA2']=pd.concat([profile(kTR(200),core,'PT200'),profile(np.where(cash,np.round(C/A,2),-5.0),core,'PCA2')],axis=1)
out['TRcore']=profile(kTR(200),core,'PT200')
for k,v in out.items(): v.to_pickle(S+f'/loop/a1r2/pv_{k}.pkl'); print(k,v.shape, 'nan share',round(v.iloc[:,0].isna().mean(),3))

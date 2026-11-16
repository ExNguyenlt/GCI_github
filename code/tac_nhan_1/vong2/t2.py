import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S+'/loop/a1r2')
from h2 import *
df,y,n,b,folds=setup(S)
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
tr=lambda X: pd.DataFrame(X).iloc[:n].reset_index(drop=True)
k=(C/G-1)/0.0132; kr=np.where(np.abs(k-np.round(k))<0.02,np.round(k),-1); kr[np.isclose(C,G)]=0
base_d=pd.DataFrame({'e':df.EXTF_MEAN.values,'e2':df.EXT_SOURCE_2.values,'e3':df.EXT_SOURCE_3.values,'age':df.DAYS_BIRTH.values,'inc':np.log(df.AMT_INCOME_TOTAL.values),
     'cr':np.log(C),'male':(df.CODE_GENDER=='M').values.astype(float),'ins':(C>G+1).astype(float),'emp':df.DAYS_EMPLOYED.values,'e1p':df.EXT_SOURCE_1_PRED.values,
     'e3na':df.EXT_SOURCE_3.isna().values.astype(float),'hi_edu':df.NAME_EDUCATION_TYPE.astype(str).str.contains('Higher').values.astype(float),'rate':df.RATE_HAT.values})
def profile(p,minc=30,pre='PP'):
    d=base_d.copy(); d['p']=p; s=pd.Series(p); cnt=s.map(s.value_counts()).values
    g=d.groupby('p'); prof=pd.DataFrame({f'{pre}_{c}':g[c].transform('mean').values for c in base_d.columns})
    prof[pre+'_estd']=g.e.transform('std').values; prof[pre+'_cnt']=cnt; prof[cnt<minc]=np.nan
    prof[pre+'_e_rel']=d.e.values-prof[pre+'_e'].values; return prof
keys={}
keys['CA2']=np.where(cash,np.round(C/A,2),-5.0)
keys['REGION']=df.REGION_POPULATION_RELATIVE.values
keys['INSK']=kr
keys['GOODS']=G
keys['ANNUITY']=A
keys['CREDIT']=C
keys['INCOME']=df.AMT_INCOME_TOTAL.values
keys['ORGxOCC']=(df.ORGANIZATION_TYPE.astype(str)+'|'+df.OCCUPATION_TYPE.astype(str)).values
keys['REGxORG']=(df.REGION_POPULATION_RELATIVE.astype(str)+'|'+df.ORGANIZATION_TYPE.astype(str)).values
keys['TERMxRATE']=np.where(cash,df.TERM_HAT.values*1000+np.round(df.RATE_HAT.values*200),-5)
keys['CA2xREGRATING']=keys['CA2']*10+df.REGION_RATING_CLIENT.values
keys['BIRTHYEAR_GENDER']=np.round(df.DAYS_BIRTH.values/365.25)*2+(df.CODE_GENDER=='M').values
P={}
for nm,p in keys.items():
    p=pd.Series(p).fillna(-9).values
    P[nm]=profile(p,pre='P_'+nm); s=pd.Series(p)
    print(f'{nm:18s} levels {s.nunique():6d}  profile:',gain(y,b,folds,tr(P[nm])),' TE:',gain(y,b,folds,pd.DataFrame(index=range(n)),tefun=te_nested(p[:n],y)),flush=True)
# ket hop
print('CA2 profile + TE',gain(y,b,folds,tr(P['CA2']),tefun=te_nested(keys['CA2'][:n],y)))
allp=pd.concat([P[k] for k in ['CA2','GOODS','ANNUITY','CREDIT','TERMxRATE','REGION']],axis=1)
print('profiles CA2+GOODS+ANNUITY+CREDIT+TERMxRATE+REGION',gain(y,b,folds,tr(allp)))
allp.to_pickle(S+'/loop/a1r2/prof_all.pkl'); P['CA2'].to_pickle(S+'/loop/a1r2/prof_ca2.pkl')
# ablation cot trong CA2
for c in P['CA2'].columns: print('  only',c,gain(y,b,folds,tr(P['CA2'][[c]]),rounds=(100,)))

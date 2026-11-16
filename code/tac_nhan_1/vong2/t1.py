import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S+'/loop/a1r2')
from h2 import *
df,y,n,b,folds=setup(S); print('base',round(AUC(y,b),5))
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
tr=lambda X: pd.DataFrame(X).iloc[:n].reset_index(drop=True)
rng=np.random.default_rng(0)
print('noise x5',gain(y,b,folds,pd.DataFrame(rng.normal(size=(n,5)))))
print('raw C/A, C/G, RATE_HAT, TERM_HAT',gain(y,b,folds,tr(df[['CREDIT_ANNUITY','CREDIT_GOODS','RATE_HAT','TERM_HAT']])))
print('EXT fills',gain(y,b,folds,tr(df[['EXT_SOURCE_1_FILL','EXT_SOURCE_2_FILL','EXT_SOURCE_3_FILL','EXTF_MEAN']])))
# product ids
pid={ 'CA2':np.round(C/A,2),'CA3':np.round(C/A,3),'CA1':np.round(C/A,1),'AC4':np.round(A/C,4)}
k=(C/G-1)/0.0132; kr=np.where(np.abs(k-np.round(k))<0.02,np.round(k),-1); kr[np.isclose(C,G)]=0
pid['CA2xK']=pid['CA2']*100+kr
for nm,p in pid.items():
    p=np.where(cash,p,-5.0); s=pd.Series(p); cnt=s.map(s.value_counts()).values
    print(nm,'levels',s.nunique(),'rows in levels>=50:',round(np.mean(cnt>=50),3),'>=200:',round(np.mean(cnt>=200),3))
    print('   TE nested (m=50):',gain(y,b,folds,pd.DataFrame(index=range(n)),tefun=te_nested(p[:n],y)))
    # target-free profile cua san pham (train+test)
    d=pd.DataFrame({'p':p,'e':df.EXTF_MEAN.values,'e2':df.EXT_SOURCE_2.values,'e3':df.EXT_SOURCE_3.values,'age':df.DAYS_BIRTH.values,'inc':np.log(df.AMT_INCOME_TOTAL.values),
                    'cr':np.log(C),'male':(df.CODE_GENDER=='M').values.astype(float),'ins':(C>G+1).astype(float),'emp':df.DAYS_EMPLOYED.values,'e1p':df.EXT_SOURCE_1_PRED.values})
    g=d.groupby('p'); prof=pd.DataFrame({f'PP_{c}':g[c].transform('mean').values for c in d.columns if c!='p'})
    prof['PP_estd']=g.e.transform('std').values; prof['PP_cnt']=cnt
    prof[cnt<30]=np.nan
    prof['PP_e_rel']=d.e.values-prof.PP_e.values
    print('   profile target-free:',gain(y,b,folds,tr(prof)))
    if nm=='CA2': prof.to_pickle(S+'/loop/a1r2/prof_ca2.pkl')
    if nm=='CA2xK': prof.add_suffix('_K').to_pickle(S+'/loop/a1r2/prof_ca2k.pkl')

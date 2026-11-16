import pandas as pd, numpy as np
D='/home/user/GCI_github/com/'
def load():
    tr=pd.concat([pd.read_csv(D+'train_part1.csv'),pd.read_csv(D+'train_part2.csv')],ignore_index=True)
    te=pd.read_csv(D+'test.csv')
    for d in (tr,te): d.columns=[c.replace('﻿','') for c in d.columns]
    y=tr.pop('TARGET').values; n=len(tr); df=pd.concat([tr,te],ignore_index=True)
    df['DAYS_EMPLOYED_ANOM']=(df.DAYS_EMPLOYED==365243).astype(int); df.loc[df.DAYS_EMPLOYED==365243,'DAYS_EMPLOYED']=np.nan
    E=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']; e=df[E]
    df['EXT_MEAN']=e.mean(1); df['EXT_STD']=e.std(1); df['EXT_PROD']=e.prod(1,min_count=3); df['EXT_MIN']=e.min(1); df['EXT_MAX']=e.max(1)
    df['EXT_W']=df.EXT_SOURCE_1.fillna(df.EXT_MEAN)*2+df.EXT_SOURCE_2.fillna(df.EXT_MEAN)+df.EXT_SOURCE_3.fillna(df.EXT_MEAN)*3
    df['EXT_NA']=e.isna().sum(1); df['EXT_HMEAN']=3/(1/e).sum(1,min_count=1)
    for a in E: df[a+'_x_AGE']=df[a]*df.DAYS_BIRTH
    df['E1E2']=df.EXT_SOURCE_1*df.EXT_SOURCE_2; df['E2E3']=df.EXT_SOURCE_2*df.EXT_SOURCE_3; df['E1E3']=df.EXT_SOURCE_1*df.EXT_SOURCE_3
    df['CREDIT_ANNUITY']=df.AMT_CREDIT/df.AMT_ANNUITY; df['CREDIT_GOODS']=df.AMT_CREDIT/df.AMT_GOODS_PRICE
    df['GOODS_MINUS']=df.AMT_GOODS_PRICE-df.AMT_CREDIT
    df['ANNUITY_INCOME']=df.AMT_ANNUITY/df.AMT_INCOME_TOTAL; df['CREDIT_INCOME']=df.AMT_CREDIT/df.AMT_INCOME_TOTAL
    df['INC_PER_PERSON']=df.AMT_INCOME_TOTAL/df.CNT_FAM_MEMBERS; df['INC_PER_CHILD']=df.AMT_INCOME_TOTAL/(1+df.CNT_CHILDREN)
    for c in ['DAYS_EMPLOYED','OWN_CAR_AGE','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION']: df[c+'_R_BIRTH']=df[c]/df.DAYS_BIRTH
    df['EXT_x_CA']=df.EXT_MEAN*df.CREDIT_ANNUITY; df['EXT_d_CA']=df.EXT_MEAN/df.CREDIT_ANNUITY
    df['ANN_LEFT']=df.AMT_INCOME_TOTAL-df.AMT_ANNUITY
    df['REQ_SUM']=df[[c for c in df.columns if c.startswith('AMT_REQ')]].sum(1)
    # term: credit/annuity rounded (loan length proxy) & implied rate
    df['TERM_ROUND']=np.round(df.CREDIT_ANNUITY)
    cats=df.select_dtypes(exclude='number').columns.tolist()
    for c in cats:
        df[c]=df[c].fillna('NA'); df[c+'_FREQ']=df[c].map(df[c].value_counts(normalize=True))
    for c in ['ORGANIZATION_TYPE','OCCUPATION_TYPE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE']:
        g=df.groupby(c)
        for v in ['EXT_MEAN','AMT_INCOME_TOTAL','CREDIT_ANNUITY','DAYS_BIRTH','ANNUITY_INCOME']:
            df[f'{c}_{v}_DIFF']=df[v]-g[v].transform('mean')
    df['INC_ORG_OCC_REL']=df.AMT_INCOME_TOTAL/df.groupby(['ORGANIZATION_TYPE','OCCUPATION_TYPE']).AMT_INCOME_TOTAL.transform('median')
    df=df.replace([np.inf,-np.inf],np.nan)
    ids=df.SK_ID_CURR.values; df=df.drop(columns='SK_ID_CURR')
    return df,y,n,cats,ids

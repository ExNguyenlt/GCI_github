# Kiem o tang meta: TE long fold cua PRODUCT x (bien khac) — mo phong "feature combination" cua CatBoost
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
from fe_r2 import PRODUCT_KEY
df=pd.read_pickle(S+'/r2_df.pkl')[0]; P=pd.Series(PRODUCT_KEY(df))
q=lambda c,k: pd.qcut(df[c],k,labels=False,duplicates='drop').fillna(-1).astype(int).astype(str)
K={'PRODUCT':P,'P x GENDER':P+'|'+df.CODE_GENDER.astype(str),'P x EDU':P+'|'+df.NAME_EDUCATION_TYPE.astype(str),'P x INCOME_TYPE':P+'|'+df.NAME_INCOME_TYPE.astype(str),
 'P x FAMILY':P+'|'+df.NAME_FAMILY_STATUS.astype(str),'P x REGION_RATING':P+'|'+df.REGION_RATING_CLIENT.astype(str),'P x EXTF3':P+'|'+q('EXTF_MEAN',3),'P x EXTF5':P+'|'+q('EXTF_MEAN',5),
 'P x AGE4':P+'|'+q('DAYS_BIRTH',4),'P x INS':P+'|'+(df.AMT_CREDIT>df.AMT_GOODS_PRICE+1).astype(str),'P x OCC':P+'|'+df.OCCUPATION_TYPE.astype(str),'P x HOUSING':P+'|'+df.NAME_HOUSING_TYPE.astype(str),
 'P x EXT3NA':P+'|'+df.EXT_SOURCE_3.isna().astype(str),'P x CREDITq5':P+'|'+q('AMT_CREDIT',5),'P x EMPq4':P+'|'+q('DAYS_EMPLOYED',4),
 'REV: AMT_CREDIT':pd.Series(np.where(df.NAME_CONTRACT_TYPE=='Cash loans','cash',df.AMT_CREDIT.astype(str))),
 'GENDER x EDU x EXTF5':df.CODE_GENDER.astype(str)+'|'+df.NAME_EDUCATION_TYPE.astype(str)+'|'+q('EXTF_MEAN',5)}
for base,nm in [(['lgbte_r2','cat_r2','lrbin_r2'],'SEL-xgb'),(['lgb_r2','xgb_r2','lrbin_r2'],'KHONG cat, KHONG TE')]:
    ref=rep('MOC '+nm,stack(base))
    for k,v in K.items():
        for al in (10,30):
            rep(f'  + TE {k} a={al}',stack(base,[te_extra(v.values,al)]),ref)

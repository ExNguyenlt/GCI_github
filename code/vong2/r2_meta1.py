# Kiem tra o tang meta tren OOF vong 1: L2.1 (TE PRODUCT), L1.2b (KNN-500 lam cot meta), L2.4a (TE nhom INCOME x EDU x tuoi 5 nam)
import sys, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
df,y_,n_,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
from fe_r2 import PRODUCT_KEY
PROD=PRODUCT_KEY(df); age5=(-df.DAYS_BIRTH//(365.25*5)).astype(int).astype(str)
GRP=(df.NAME_INCOME_TYPE.astype(str)+'|'+df.NAME_EDUCATION_TYPE.astype(str)+'|'+age5).values
GRP2=(df.NAME_INCOME_TYPE.astype(str)+'|'+df.NAME_EDUCATION_TYPE.astype(str)+'|'+age5+'|'+df.CODE_GENDER.astype(str)+'|'+df.OCCUPATION_TYPE.astype(str)).values
KN=pd.read_pickle(S+'/r1_knn.pkl'); fidx={tuple(b[:5]):i for i,(a,b) in enumerate(folds)}
def knn_extra(col):
    def cv(a,b): k=KN[fidx[tuple(b[:5])]]; return lg(k['tr'][col].values),lg(k['va'][col].values)
    return cv,None
N4=['lgb_r1','cat_r1','xgb_r1','lrbin_r1']
ref=rep('stack r1 (moc)',stack(N4))
for al in (5,10,20): rep(f'L2.1 + TE PRODUCT a={al}',stack(N4,[te_extra(PROD,al)]),ref)
rep('L1.2b + KNN1_500',stack(N4,[knn_extra('KNN1_500')]),ref)
rep('L1.2b + KNN2_500',stack(N4,[knn_extra('KNN2_500')]),ref)
rep('L1.2b + KNN1_500+KNN2_500',stack(N4,[knn_extra('KNN1_500'),knn_extra('KNN2_500')]),ref)
rep('L1.2b + KNN1_50',stack(N4,[knn_extra('KNN1_50')]),ref)
rep('L2.4a + TE INC x EDU x AGE5 a=20',stack(N4,[te_extra(GRP,20)]),ref)
rep('L2.4a + TE INCxEDUxAGE5xGENDERxOCC a=20',stack(N4,[te_extra(GRP2,20)]),ref)
r2=rep('TE PRODUCT a=10 + KNN1_500',stack(N4,[te_extra(PROD,10),knn_extra('KNN1_500')]),ref)
rep('TE PRODUCT a=10 + TE GRP',stack(N4,[te_extra(PROD,10),te_extra(GRP,20)]),ref)

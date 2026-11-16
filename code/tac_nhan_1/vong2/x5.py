import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S+'/loop/a1r2')
from harness import base,gain,oe
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y); tr=df.iloc[:n].reset_index(drop=True)
b=base(S); from sklearn.metrics import roc_auc_score as AUC; print('base auc',AUC(y,b))
cash=(tr.NAME_CONTRACT_TYPE=='Cash loans').values
def lvl(x,ms=(4500,9000,22500,45000,225000)):
    out=np.zeros(len(x),int)
    for i,m in enumerate(ms): out[np.mod(x,m)==0]=i+1
    out[np.mod(x,1)!=0]=-1; return out
F={}
F['ANN_ROUND']=np.where(cash,lvl(tr.AMT_ANNUITY.values),99)
F['INC_ROUND']=lvl(tr.AMT_INCOME_TOTAL.values)
F['GOODS_ROUND']=np.where(cash,lvl(tr.AMT_GOODS_PRICE.values),99)
F['GOODS_MOD45k']=np.where(cash,np.round(np.mod(tr.AMT_GOODS_PRICE.values,45000)/4500),99)
for k,v in F.items():
    print('==',k); print(oe(y,b,v).to_string()); print('gain',gain(y,b,pd.DataFrame({k:v})))

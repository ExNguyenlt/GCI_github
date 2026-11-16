# kiem tra "thong tin con lai": feature co giai thich them phan du cua stack r1 khong (CV LR tren logit OOF + one-hot dac trung)
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def base(S):
    m=[lg(np.load(S+f'/oof_{k}_r1.npy')) for k in ['lgb','cat','xgb']]
    b=np.mean(m,0)*0.85+0.15*lg(np.load(S+'/oof_lrbin_r1.npy')); return b
def gain(y,b,F):
    """F: DataFrame cac cot (categorical se one-hot); tra ve (auc_base, auc_new, chenh tung fold)"""
    X=pd.get_dummies(F.astype(str),drop_first=True).values.astype(float)
    pb=np.zeros(len(y)); pn=np.zeros(len(y)); d=[]
    for a,v in StratifiedKFold(5,shuffle=True,random_state=42).split(b,y):
        m0=LogisticRegression(C=1.0,max_iter=500).fit(b[a,None],y[a]); pb[v]=m0.decision_function(b[v,None])
        Z=np.c_[b,X]; m1=LogisticRegression(C=0.1,max_iter=2000).fit(Z[a],y[a]); pn[v]=m1.decision_function(Z[v])
        d.append(AUC(y[v],pn[v])-AUC(y[v],pb[v]))
    return round(AUC(y,pb),5),round(AUC(y,pn),5),np.round(d,5)
def oe(y,b,f):
    p=1/(1+np.exp(-b)); t=pd.DataFrame({'f':f,'y':y,'p':p}).groupby('f').agg(n=('y','size'),obs=('y','mean'),exp=('p','mean'))
    t['o/e']=t.obs/t.exp; return t[t.n>=200].round(4)

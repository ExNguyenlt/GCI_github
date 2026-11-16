# Harness 2: LGB boosting tu logit stack r1 (init_score) tren nhom dac trung ung vien -> thong tin CON LAI ma stack chua nam.
import numpy as np,pandas as pd,lightgbm as lgb,sys
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score as AUC
sys.path.insert(0,sys.path[0]); from harness import base
def setup(S):
    df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y); b=base(S)
    # hieu chuan b ve logit that (LR 1 bien) de init_score dung thang
    from sklearn.linear_model import LogisticRegression
    m=LogisticRegression(C=100).fit(b[:,None],y); b=m.decision_function(b[:,None])
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(b,y))
    return df,y,n,b,folds
P=dict(objective='binary',learning_rate=0.03,max_depth=3,num_leaves=8,min_child_samples=200,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=2,seed=0)
def gain(y,b,folds,F,rounds=(100,300),tefun=None):
    """F: DataFrame (n dong train). tefun(a_idx, v_idx)->(Ftr_extra, Fva_extra) cho dac trung dung nhan (long fold)."""
    out={}
    for R in rounds: out[R]=[]
    for a,v in folds:
        Xa=F.iloc[a].reset_index(drop=True); Xv=F.iloc[v].reset_index(drop=True)
        if tefun is not None:
            ea,ev=tefun(a,v); Xa=pd.concat([Xa,ea],axis=1); Xv=pd.concat([Xv,ev],axis=1)
        m=lgb.train(P,lgb.Dataset(Xa,y[a],init_score=b[a]),max(rounds))
        for R in rounds:
            p=b[v]+m.predict(Xv,num_iteration=R,raw_score=True); out[R].append(AUC(y[v],p)-AUC(y[v],b[v]))
    return {R:(round(float(np.mean(d)),5),int((np.array(d)>0).sum())) for R,d in out.items()}
def te_nested(key,y,m=50,inner=5):
    """tra ve tefun: target-encoding lam min, long fold"""
    key=np.asarray(key)
    def enc(ktr,ytr,kq):
        s=pd.DataFrame({'k':ktr,'y':ytr}).groupby('k').y.agg(['sum','count']); pr=ytr.mean()
        t=(s['sum']+m*pr)/(s['count']+m); return pd.Series(kq).map(t).fillna(pr).values
    def f(a,v):
        ea=np.zeros(len(a))
        for i,j in KFold(inner,shuffle=True,random_state=1).split(a): ea[j]=enc(key[a[i]],y[a[i]],key[a[j]])
        return pd.DataFrame({'TE':ea}),pd.DataFrame({'TE':enc(key[a],y[a],key[v])})
    return f

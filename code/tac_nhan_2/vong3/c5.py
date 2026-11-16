# (1) harness "thong tin con lai": LGB nong boosting tiep tu logit stack r2 tren TOAN BO cot mo hinh; (2) LGB 5 fold x 1 seed lr 0.05: TE / PROF voi khoa PRODUCT x CREDIT q5
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import PRODUCT_KEY, nested_te, profile
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl')
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; COLS=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']
PROD=PRODUCT_KEY(df); cq=pd.qcut(df.AMT_CREDIT,5,labels=False).astype(str).values; PC=np.char.add(np.char.add(PROD.astype(str),'|'),cq)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
kTR=np.where(cash,df.TERM_HAT.values*100000+np.round(df.RATE_HAT.values*200),-5); kTR=pd.Series(kTR).fillna(-9).values
core=pd.DataFrame({'e':df.EXTF_MEAN.values,'e3':df.EXT_SOURCE_3.values,'age':df.DAYS_BIRTH.values,'emp':df.DAYS_EMPLOYED.values})
p,_=profile(kTR*10+pd.qcut(df.AMT_CREDIT,5,labels=False).values,core,'PQ',std=False)
for c in p.columns: df[c]=p[c].values
PQ=list(p.columns)
X=df.iloc[:n].reset_index(drop=True); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
if sys.argv[1]=='H':
    s=np.load(S+'/oof_final_r2.npy'); b=LogisticRegression(C=100).fit(s[:,None],y).decision_function(s[:,None]); print('stack',round(A(y,b),5))
    P=dict(objective='binary',learning_rate=0.03,max_depth=3,num_leaves=8,min_child_samples=200,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,lambda_l2=20,verbose=-1,n_jobs=2,seed=0)
    for nm,cols in [('ALL model cols',COLS),('PQ (ho so khoa TRxCREDITq5)',PQ),('noise5',None)]:
        F=X[cols] if cols else pd.DataFrame(np.random.RandomState(0).randn(n,5)); out={50:[],100:[],300:[]}
        for a,v in folds:
            m=lgb.train(P,lgb.Dataset(F.iloc[a],y[a],init_score=b[a]),300)
            for R in out: out[R].append(A(y[v],b[v]+m.predict(F.iloc[v],num_iteration=R,raw_score=True))-A(y[v],b[v]))
        print(nm,{R:(round(float(np.mean(d)),5),int((np.array(d)>0).sum())) for R,d in out.items()},flush=True)
else:
    P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=2,metric='auc',seed=0)
    def run(cols,sd):
        res=[]
        for f,(a,b) in enumerate(folds):
            Xa=X.loc[a,cols]; Xb=X.loc[b,cols]
            m=lgb.train(dict(P,seed=sd),lgb.Dataset(Xa,y[a]),5000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(Xb,num_iteration=m.best_iteration)))
        return np.array(res)
    for sd in (1,2):
        r0=run(COLS,sd); r1=run(COLS+PQ,sd); d=r1-r0
        print('seed',sd,'PROF',np.round(r0,5),round(r0.mean(),5),'PROF+PQ',np.round(r1,5),round(r1.mean(),5),'d',np.round(d,5),'mean %+.5f pos %d/5'%(d.mean(),(d>0).sum()),flush=True)

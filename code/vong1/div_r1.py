# Mo hinh khac ho de da dang ensemble: (1) LogisticRegression tren one-hot bin quantile (GAM cong tinh), (2) sklearn MLP tren quantile-normal.
# python3 -I div_r1.py <lrbin|mlp> <useknn 0/1> <tag>
import sys, time, numpy as np, pandas as pd, scipy.sparse as sp, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
from sklearn.linear_model import LogisticRegression
which,useknn,tag=sys.argv[1],int(sys.argv[2]),sys.argv[3]
df,y,n,cats,ids,G=load_r1()
newc=set(sum(G.values(),[])); num=[c for c in df.columns if c not in newc and c not in cats]+G['IMP']+G['RATE']+G['TERM']+G['DATE']
KN=pd.read_pickle(S+'/r1_knn.pkl') if useknn else None; KC=['KNN1_500','KNN2_500']
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
NB=16
if which=='lrbin':
    blocks=[]
    for c in num:
        x=df[c].values.astype(float); ok=~np.isnan(x)
        if np.unique(x[ok]).size<=NB: codes=pd.factorize(np.where(ok,x,-9e9))[0]
        else:
            qs=np.unique(np.nanquantile(x,np.linspace(0,1,NB+1)[1:-1])); codes=np.searchsorted(qs,x); codes[~ok]=len(qs)+1
        blocks.append(pd.get_dummies(codes.astype(str),prefix=c,sparse=False).astype(np.float32).values)
    for c in cats: blocks.append(pd.get_dummies(df[c].astype(str)).astype(np.float32).values)
    M=sp.csr_matrix(np.hstack(blocks)); print('design',M.shape,flush=True)
    Xtr,Xte=M[:n],M[n:]
else:
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.neural_network import MLPClassifier
    Z=df[num].copy(); na=Z.isna(); Z=Z.fillna(Z.median())
    Z=QuantileTransformer(n_quantiles=500,output_distribution='normal',subsample=10**9,random_state=0).fit_transform(Z.values)
    nac=na.columns[na.mean()>0.01]; Z=np.hstack([Z,na[nac].values.astype(float)]+[pd.get_dummies(df[c]).astype(float).values for c in cats])
    Xtr,Xte=Z[:n].astype(np.float32),Z[n:].astype(np.float32); print('design',Z.shape,flush=True)
def addk(Xm,K):
    if K is None: return Xm
    K=np.log(K.values/(1-K.values)+1e-9) if False else (K.values-0.08)*10
    return sp.hstack([Xm,sp.csr_matrix(K)]).tocsr() if sp.issparse(Xm) else np.hstack([Xm,K])
oof=np.zeros(n); pt=np.zeros(len(df)-n); t0=time.time()
for f,(a,b) in enumerate(folds):
    Xa,Xb,Tt=Xtr[a],Xtr[b],Xte
    if useknn: Xa,Xb,Tt=addk(Xa,KN[f]['tr'][KC]),addk(Xb,KN[f]['va'][KC]),addk(Tt,KN[f]['te'][KC])
    if which=='lrbin':
        m=LogisticRegression(C=0.05,solver='lbfgs',max_iter=2000); m.fit(Xa,y[a])
        p=m.predict_proba(Xb)[:,1]; q=m.predict_proba(Tt)[:,1]
    else:
        ps=[];qs=[]
        for s in (0,1):
            m=MLPClassifier((128,64),alpha=1e-3,batch_size=512,learning_rate_init=1e-3,max_iter=40,early_stopping=True,validation_fraction=0.1,n_iter_no_change=5,random_state=s)
            m.fit(Xa,y[a]); ps.append(m.predict_proba(Xb)[:,1]); qs.append(m.predict_proba(Tt)[:,1])
        p=np.mean(ps,0); q=np.mean(qs,0)
    oof[b]=p; pt+=q/5; print(tag,f,round(A(y[b],p),5),f'{time.time()-t0:.0f}s',flush=True)
print(tag,'OOF AUC',round(A(y,oof),5),'folds',[round(A(y[b],oof[b]),5) for a,b in folds],flush=True)
np.save(f'{S}/oof_{tag}_r1.npy',oof); np.save(f'{S}/pt_{tag}_r1.npy',pt)

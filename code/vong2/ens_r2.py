# Vong 2: ensemble. Stacking LR tren logit (CV long nhau tren fold rs=42) + cot phu o meta (TE PRODUCT long nghiem ngat R14, KNN, TE nhom).
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import te_map, nested_te
y=np.load(S+'/y.npy'); n=len(y); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def ld(m,k): return np.load(f'{S}/{k}_{m}.npy')
def fa(o): return np.array([A(y[b],o[b]) for a,b in folds])
def te_extra(keys,alpha=10):
    """keys: mang khoa cho train+test (n + n_test). Tra ve (cv_fn, full_fn)."""
    ktr=keys[:n]; idx=np.arange(n)
    def cv(a,b):
        ta,(tb,)=nested_te(ktr,y,a,[b],alpha); return lg(ta),lg(tb)
    def full():
        ta,_=nested_te(ktr,y,idx,[],alpha); pr=y.mean(); d=pd.DataFrame({'k':ktr,'y':y}).groupby('k').y.agg(['sum','count'])
        v=(d['sum']+alpha*pr)/(d['count']+alpha); return lg(ta),lg(pd.Series(keys[n:]).map(v).fillna(pr).values)
    return cv,full
def stack(names,extras=(),C=1.0,test=False):
    """extras: danh sach (cv_fn, full_fn). Tra ve oof (meta cham tren fold chua thay), [du doan test, he so] neu test=True."""
    L=np.column_stack([lg(ld(m,'oof')) for m in names]); oof=np.zeros(n)
    for a,b in folds:
        ex=[e[0](a,b) for e in extras]; Fa=np.column_stack([L[a]]+[e[0] for e in ex]); Fb=np.column_stack([L[b]]+[e[1] for e in ex])
        mu,sd=Fa.mean(0),Fa.std(0)+1e-9; oof[b]=LogisticRegression(C=C,max_iter=2000).fit((Fa-mu)/sd,y[a]).decision_function((Fb-mu)/sd)
    if not test: return oof
    Pt=np.column_stack([lg(ld(m,'pt')) for m in names]); ex=[e[1]() for e in extras]
    Fa=np.column_stack([L]+[e[0] for e in ex]); Ft=np.column_stack([Pt]+[e[1] for e in ex]); mu,sd=Fa.mean(0),Fa.std(0)+1e-9
    m=LogisticRegression(C=C,max_iter=2000).fit((Fa-mu)/sd,y)
    return oof,m.decision_function((Ft-mu)/sd),np.round(m.coef_.ravel(),3)
def rankavg(names): return np.mean([rankdata(ld(m,'oof'))/n for m in names],0), np.mean([rankdata(ld(m,'pt'))/len(ld(m,'pt')) for m in names],0)
def rep(tag,o,ref=None):
    f=fa(o); s=f'{tag:44s} OOF {A(y,o):.5f} folds {np.round(f,5)}'
    if ref is not None: d=f-ref; s+=f' | d {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5'
    print(s,flush=True); return f

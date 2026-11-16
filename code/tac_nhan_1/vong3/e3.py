# Tu chung cat (self-distillation / nhan mem): inner 5-fold tren a -> p_oof ; hoc lai voi nhan (1-al)*y + al*p_oof, objective cross_entropy
from base import *
INN={}
def inner(fo):
    f=f'inner_{fo}.npy'
    if os.path.exists(f): return np.load(f)
    a,b=folds[fo]; X=df[BASE].iloc[:n].reset_index(drop=True); Xa=X.iloc[a].reset_index(drop=True); p=np.zeros(len(a))
    for ia,ib in StratifiedKFold(5,shuffle=True,random_state=1).split(a,y[a]):
        for s in (0,1):
            m=fit(Xa.iloc[ia],y[a][ia],Xa.iloc[ib],y[a][ib],s,P2=dict(learning_rate=0.1)); p[ib]+=m.predict(Xa.iloc[ib],num_iteration=m.best_iteration)/2
    print('inner',fo,A(y[a],p),flush=True); np.save(f,p); return p
def mk(al):
    def hook(a,b,Xa,Xb,s):
        fo=[i for i in (0,1) if len(folds[i][0])==len(a) and folds[i][0][0]==a[0] and (folds[i][0]==a).all()][0]
        p=inner(fo); return Xa,Xb,(1-al)*y[a]+al*p,None
    return hook
for al in (0.3,0.6,0.9):
    screen(f'distill_{al}',P2=dict(objective='cross_entropy',metric='auc'),hook=mk(al))
screen('xent_only',P2=dict(objective='cross_entropy',metric='auc'))

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
from scipy.stats import rankdata
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad/'
y=np.load(S+'y.npy'); M=['lgb','cat','xgb','lgb_opt','cat_opt','xgb_opt']
O={m:np.load(S+f'oof_{m}.npy') for m in M}; P={m:np.load(S+f'pt_{m}.npy') for m in M}
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
R={m:rankdata(O[m])/len(y) for m in M}
O['rank6']=np.mean([R[m] for m in M],0)
print('per-fold AUC')
for m in list(O):
    fa=[A(y[b],O[m][b]) for _,b in folds]; print(f'{m:9s}', ' '.join(f'{v:.4f}' for v in fa), f'mean {np.mean(fa):.5f} std {np.std(fa):.4f} oof {A(y,O[m]):.5f}')
print('Spearman corr OOF'); print(pd.DataFrame({m:R[m] for m in M}).corr().round(4))
print('Spearman corr test'); print(pd.DataFrame({m:rankdata(P[m]) for m in M}).corr().round(4))
# paired fold diffs vs lgb
for m in M[1:]+['rank6']:
    d=[A(y[b],O[m][b])-A(y[b],O['lgb'][b]) for _,b in folds]; print('diff vs lgb',m,np.round(d,4),'mean',round(np.mean(d),5),'std',round(np.std(d),5))
# bootstrap SE of AUC
rng=np.random.default_rng(0); bs=[]; bd=[]
for i in range(200):
    ix=rng.integers(0,len(y),len(y)); bs.append(A(y[ix],O['lgb'][ix])); bd.append(A(y[ix],O['rank6'][ix])-A(y[ix],O['lgb'][ix]))
print('bootstrap SE lgb AUC',np.std(bs).round(5),' rank6-lgb diff mean',np.mean(bd).round(5),'SE',np.std(bd).round(5))

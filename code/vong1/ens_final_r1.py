# Vong 1: so sanh cac cach ensemble (rank-avg / stacking LR, CV long nhau tren fold rs=42) va ghi submission.
# python3 -I ens_final_r1.py [write <set_name>]
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'
y=np.load(S+'/y.npy'); n=len(y); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def ld(m,k): return np.load(f'{S}/{k}_{m}.npy')
def stack(names):
    O=np.column_stack([lg(ld(m,'oof')) for m in names]); P=np.column_stack([lg(ld(m,'pt')) for m in names])
    mu,sd=O.mean(0),O.std(0); O=(O-mu)/sd; P=(P-mu)/sd; oof=np.zeros(n)
    for a,b in folds: oof[b]=LogisticRegression(C=1.0,max_iter=1000).fit(O[a],y[a]).decision_function(O[b])  # meta cham tren fold chua thay
    m=LogisticRegression(C=1.0,max_iter=1000).fit(O,y); return oof,m.decision_function(P),dict(zip(names,np.round(m.coef_.ravel(),3)))
def rankavg(names):
    return (np.mean([rankdata(ld(m,'oof'))/n for m in names],0), np.mean([rankdata(ld(m,'pt'))/len(ld(m,'pt')) for m in names],0), None)
def fa(o): return np.array([A(y[b],o[b]) for a,b in folds])
R0=['lgb','cat','xgb','lgb_opt','cat_opt','xgb_opt']; G1=['lgb_r1','cat_r1','xgb_r1']; D1=['lrbin_r1','mlp_r1']
ref=fa(stack(R0)[0])
SETS={'rank_G1':(rankavg,G1),'stack_G1':(stack,G1),
      'rank_G1+D1':(rankavg,G1+D1),'stack_G1+D1':(stack,G1+D1),
      'stack_G1+lrbin':(stack,G1+['lrbin_r1']),'stack_G1+mlp':(stack,G1+['mlp_r1']),
      'stack_lgbcat+D1':(stack,['lgb_r1','cat_r1']+D1),
      'stack_R0+G1+D1':(stack,R0+G1+D1)}
if __name__=='__main__':
    for m in R0+G1+D1:
        o=ld(m,'oof'); print(f'{m:10s} OOF {A(y,o):.5f} folds {np.round(fa(o),5)}')
    rg=rankavg(G1)[0]
    print('spearman vs rank_G1:',{m:round(spearmanr(ld(m,'oof'),rg)[0],4) for m in G1+D1})
    res={}
    for k,(f,names) in SETS.items():
        o,p,c=f(names); res[k]=(o,p); d=fa(o)-ref
        print(f'{k:16s} OOF {A(y,o):.5f} folds {np.round(fa(o),5)} vs R0stack {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5', c if c else '')
    b=fa(res['stack_G1+D1'][0]);
    for k in ['rank_G1','stack_G1','rank_G1+D1','stack_G1+lrbin','stack_G1+mlp','stack_lgbcat+D1','stack_R0+G1+D1']:
        d=fa(res[k][0])-b; print(f'  {k:16s} - stack_G1+D1: {np.round(d,5)} mean {d.mean():+.5f}')
    if len(sys.argv)>2 and sys.argv[1]=='write':
        k=sys.argv[2]; p=res[k][1]; ids=np.load(S+'/ids.npy')
        ss=pd.read_csv('/home/user/GCI_github/com/sample_submission.csv'); ss.columns=[c.replace('﻿','') for c in ss.columns]
        assert len(ss)==len(p)==len(ids) and (ss.SK_ID_CURR.values==ids).all()
        pr=rankdata(p)/len(p)   # AUC chi phu thuoc thu hang -> dua ve (0,1]
        sub=pd.DataFrame({'SK_ID_CURR':ss.SK_ID_CURR.values,'TARGET':pr}); assert sub.TARGET.notna().all()
        sub.to_csv(S+'/loop/sub_round1.csv',index=False); print('wrote',S+'/loop/sub_round1.csv',k,sub.shape)

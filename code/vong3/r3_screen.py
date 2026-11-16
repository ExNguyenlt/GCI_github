# Sang loc R3 vong 3 (LGB lr 0.05 cay nong, fold 0/1 x seed 0/1, ghep cap; BASE = IMP+RATE+TERM+PROF nhu r2_screen '+PROF').
# python3 -I r3_screen.py <variant> ...   ; ket qua -> r3_screen.json
#  BASE | PQ | T_soft35 | T_hard40 | T_soft35+PQ | PS50 | ET | LT | DART | NI<thr> (null importance trong fold) | PQ+PS50
import sys, os, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r3 import load_r3, term_alt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r3()
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]
BASE=orig+G['IMP']+G['RATE']+G['TERM']+G['PROF']
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
JS=S+'/r3_screen.json'
FS=[int(x) for x in os.environ.get('FS','0,1').split(',')]; SS=[int(x) for x in os.environ.get('SS','0,1').split(',')]; SUF=os.environ.get('SUF','')
def load(): return json.load(open(JS)) if os.path.exists(JS) else {}
def report(name,res,ref):
    R=load(); R[name+SUF]=[float(x) for x in res]; json.dump(R,open(JS,'w'))
    msg=f'{name+SUF:26s} {np.round(res,5)} mean {np.mean(res):.5f}'
    if ref+SUF in R and ref!=name:
        d=np.array(res)-np.array(R[ref+SUF]); msg+=f' | vs {ref} {d.mean():+.5f} pos {int((d>0).sum())}/{len(d)}'
    print(msg,flush=True)
def null_imp(Xa,ya,nrun=20,rounds=150):
    """Null Importance (Grinberg): gain that / phan vi 75 cua gain khi xao tron nhan; CHI dung nhan phan train cua fold."""
    Pn=dict(P,learning_rate=0.1,seed=0); act=np.zeros(Xa.shape[1])
    for s in range(3): act+=lgb.train(dict(Pn,seed=s),lgb.Dataset(Xa,ya),rounds).feature_importance('gain')/3
    rs=np.random.RandomState(0); nul=[]
    for r in range(nrun): nul.append(lgb.train(dict(Pn,seed=r),lgb.Dataset(Xa,rs.permutation(ya)),rounds).feature_importance('gain'))
    nul=np.array(nul); return np.log((1e-10+act)/(1+np.percentile(nul,75,axis=0)))
for v in sys.argv[1:]:
    t=time.time(); D=df[BASE].copy(); P2={}; ref='BASE'; fixed=None; ni=None
    for tok in v.split('+'):
        if tok=='BASE': pass
        elif tok=='PQ': D=pd.concat([D,df[G['PQ']]],axis=1)
        elif tok.startswith('T_'):
            ta=term_alt(df,tok[2:])
            for c in ta.columns:
                if c in D.columns: D[c]=ta[c].values
            D['__pq']=0; D=D.drop(columns='__pq'); ALT=ta
        elif tok=='PQA': D=pd.concat([D,ALT[G['PQ']].set_axis(D.index)],axis=1)
        elif tok=='PS50': P2['path_smooth']=50
        elif tok=='ET': P2['extra_trees']=True
        elif tok=='LT': P2.update(linear_tree=True,linear_lambda=10.0)
        elif tok=='DART': P2.update(boosting='dart',drop_rate=0.1,skip_drop=0.5,learning_rate=0.1); fixed=600
        elif tok.startswith('NI'): ni=float(tok[2:])
        else: raise SystemExit('?'+tok)
    X=D.iloc[:n].reset_index(drop=True); res=[]; info=''
    for fo in FS:
        a,b=folds[fo]; Xa=X.iloc[a].reset_index(drop=True); Xb=X.iloc[b].reset_index(drop=True)
        if LTn:=('linear_tree' in P2):
            num=[c for c in Xa.columns if c not in cats]; med=Xa[num].median(); Xa[num]=Xa[num].fillna(med); Xb[num]=Xb[num].fillna(med)   # linear_tree khong nhan NaN tot
        if ni is not None:
            sc=null_imp(Xa,y[a]); keep=[c for c,s in zip(Xa.columns,sc) if s>ni]; info+=f' fold{fo}: giu {len(keep)}/{Xa.shape[1]} cot; bo: {[c for c in Xa.columns if c not in keep][:12]}'
            Xa=Xa[keep]; Xb=Xb[keep]
        for s in SS:
            p=dict(P,seed=s); p.update(P2)
            if fixed:
                ev={}; m=lgb.train(dict(p,metric='auc'),lgb.Dataset(Xa,y[a]),fixed,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.record_evaluation(ev)])
                res.append(A(y[b],m.predict(Xb))); c=ev['valid_0']['auc']; info+=f' [dart f{fo}s{s} auc@200/400/600 {c[199]:.5f}/{c[399]:.5f}/{c[599]:.5f}]'
            else:
                m=lgb.train(dict(p,metric='auc'),lgb.Dataset(Xa,y[a]),5000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
                res.append(A(y[b],m.predict(Xb,num_iteration=m.best_iteration)))
    report(v,res,ref); print(f'   ({time.time()-t:.0f}s, {D.shape[1]} cols){info}',flush=True)

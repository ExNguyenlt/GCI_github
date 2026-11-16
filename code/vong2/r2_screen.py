# Sang loc R3 vong 2 (LGB lr 0.05 cay nong, fold 0/1 x 2 seed, ghep cap). Dung: python3 -I r2_screen.py <variant> [<variant> ...]
# Ket qua luu dan vao r2_screen.json (bo qua bien the da co). Cu phap bien the:  [REFNAME|]+GRP+GRP-GRP-col~TE~PP~PL   (so voi REFNAME, mac dinh REF)
import sys, os, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r2 import load_r2, PRODUCT_KEY, nested_te, te_map
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
PK=S+'/r2_df.pkl'
if os.path.exists(PK): df,y,n,cats,ids,G=pd.read_pickle(PK)
else: r=load_r2(); pd.to_pickle(r,PK); df,y,n,cats,ids,G=r
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]
REF=orig+G['IMP']+G['RATE']+G['TERM']
G['RES']=[e+'_RES' for e in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
_vc=pd.Series(PRODUCT_KEY(df)).value_counts(); df['PRODUCT_CAT']=pd.Series(np.where(pd.Series(PRODUCT_KEY(df)).map(_vc).values<30,'rare',PRODUCT_KEY(df))).astype('category').values
X=df.iloc[:n].reset_index(drop=True); T=df.iloc[n:].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
PROD=PRODUCT_KEY(df); age5=(-df.DAYS_BIRTH//(365.25*5)).astype(int).astype(str)
GRPKEY=(df.NAME_INCOME_TYPE.astype(str)+'|'+df.NAME_EDUCATION_TYPE.astype(str)+'|'+age5).values
kTR=np.where((df.NAME_CONTRACT_TYPE=='Cash loans').values,df.TERM_HAT.values*100000+np.round(df.RATE_HAT.values*200),-5); kTR=pd.Series(kTR).fillna(-9).values
JS=S+'/r2_screen.json'; R=json.load(open(JS)) if os.path.exists(JS) else {}
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def colsof(spec):
    cols=list(REF)
    import re
    for sgn,nm in re.findall(r'([+-])([A-Za-z0-9_]+)',spec):
        cc=G[nm] if nm in G else [nm]
        if sgn=='+': cols+=[c for c in cc if c not in cols]
        else: cols=[c for c in cols if c not in cc]
        if nm=='IMPB' and sgn=='+': cols=[c for c in cols if c not in G['IMP']]
    return cols
def fit(Xa,ya,Xb,yb,s,w=None):
    m=lgb.train(dict(P,seed=s),lgb.Dataset(Xa,ya,weight=w),5000,valid_sets=[lgb.Dataset(Xb,yb)],callbacks=[lgb.early_stopping(200,verbose=False)]); return m
def run(v):
    if v in R: print(f'{v:40s} (cached)',np.round(R[v],5),'mean',round(np.mean(R[v]),5),flush=True); return
    body=v.split('|')[-1]; parts=body.split('~'); cols=colsof(parts[0]); flags=parts[1:]
    res=[]; t=time.time()
    if 'PCAT' in flags: cols=cols+['PRODUCT_CAT']
    for fo in (0,1):
        a,b=folds[fo]; Xa=X.loc[a,cols].reset_index(drop=True); Xb=X.loc[b,cols].reset_index(drop=True); Tt=T[cols].copy()
        if 'TE' in flags:   # TE PRODUCT long fold (R14), alpha=10
            ta,(tb,)=nested_te(PROD[:n],y,a,[b],10); Xa['TE_PRODUCT']=ta; Xb['TE_PRODUCT']=tb
        if 'TEG' in flags:  # TE (INCOME_TYPE x EDU x tuoi 5 nam) long fold
            ta,(tb,)=nested_te(GRPKEY[:n],y,a,[b],20); Xa['TE_GRP']=ta; Xb['TE_GRP']=tb
        if 'PP' in flags:   # phuong phap 2 (A1): ho so theo du doan cua mo hinh, long fold. inner 5-fold trong a; b va test = TB cac mo hinh inner
            pa=np.zeros(len(a)); pb=np.zeros(len(b)); pt=np.zeros(len(T))
            for ia,ib in StratifiedKFold(5,shuffle=True,random_state=1).split(a,y[a]):
                m=fit(Xa.iloc[ia],y[a][ia],Xa.iloc[ib],y[a][ib],0); bi=m.best_iteration
                pa[ib]=m.predict(Xa.iloc[ib],num_iteration=bi,raw_score=True); pb+=m.predict(Xb,num_iteration=bi,raw_score=True)/5; pt+=m.predict(Tt,num_iteration=bi,raw_score=True)/5
            allp=np.concatenate([pa,pb,pt]); k=np.concatenate([kTR[:n][a],kTR[:n][b],kTR[n:]]); s_=pd.Series(allp); gk=pd.Series(k)
            mean=s_.groupby(gk).transform('mean').values.copy(); cnt=gk.map(gk.value_counts()).values; mean[cnt<30]=np.nan
            Xa['PP_logit']=mean[:len(a)]; Xb['PP_logit']=mean[len(a):len(a)+len(b)]
        for s in (0,1):
            if 'PL' in flags:  # pseudo-label (L1.5a): mo hinh hoc tren a -> du doan test -> them test voi nhan mem (2 ban sao co trong so p va 1-p), hoc lai
                m0=fit(Xa,y[a],Xb,y[b],s); q=m0.predict(Tt,num_iteration=m0.best_iteration)
                if 'HARD' in flags:
                    lo,hi=np.quantile(q,[0.5,0.99]); sel=(q<=lo)|(q>=hi); XX=pd.concat([Xa,Tt[sel]],ignore_index=True); yy=np.concatenate([y[a],(q[sel]>=hi).astype(int)]); w=None
                else:
                    XX=pd.concat([Xa,Tt,Tt],ignore_index=True); yy=np.concatenate([y[a],np.ones(len(Tt),int),np.zeros(len(Tt),int)]); w=np.concatenate([np.ones(len(a)),q,1-q])
                m=fit(XX,yy,Xb,y[b],s,w)
            else: m=fit(Xa,y[a],Xb,y[b],s)
            res.append(float(A(y[b],m.predict(Xb,num_iteration=m.best_iteration))))
    R[v]=res; json.dump(R,open(JS,'w')); refn=v.split('|')[0] if '|' in v else 'REF'
    print(f'{v:40s}',np.round(res,5),'mean',round(np.mean(res),5),f'{len(cols)} cols {time.time()-t:.0f}s',flush=True)
    if refn in R and refn!=v:
        d=np.array(res)-np.array(R[refn]); print(f'      vs {refn}: {np.round(d,5)} mean {d.mean():+.5f} pos {int((d>0).sum())}/4',flush=True)
for v in sys.argv[1:]: run(v)

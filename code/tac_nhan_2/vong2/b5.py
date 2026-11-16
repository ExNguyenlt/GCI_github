# Kiem chung chat che TE "ma san pham": (a) long nhau nghiem ngat trong meta, (b) them lam dac trung LGB (sang loc fold0/1 x 2 seed), (c) uid
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from ens_final_r1 import ld, lg, folds
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1()
PRODUCT=np.round(df.AMT_CREDIT/df.AMT_ANNUITY,2).astype(str).values
TR=(df.TERM_HAT.fillna(-1).astype(str)+'_'+np.round(df.RATE_HAT*100).fillna(-1).astype(str)).values
PR3=np.round(df.AMT_CREDIT/df.AMT_ANNUITY,3).astype(str).values
def te_map(keys,yy,idx_fit,idx_apply,alpha):
    pr=yy[idx_fit].mean(); d=pd.DataFrame({'k':keys[idx_fit],'y':yy[idx_fit]}).groupby('k').y.agg(['sum','count'])
    v=(d['sum']+alpha*pr)/(d['count']+alpha); return pd.Series(keys[idx_apply]).map(v).fillna(pr).values
def nested_te(keys,a,b,alpha,seed=0):
    """TE cho phan train a bang inner 5-fold trong a; cho b (va test) tu toan bo a"""
    ta=np.zeros(len(a))
    for ia,ib in StratifiedKFold(5,shuffle=True,random_state=seed).split(a,y[a]): ta[ib]=te_map(keys,y,a[ia],a[ib],alpha)
    return ta, te_map(keys,y,a,b,alpha)
def fa(o): return np.array([A(y[b],o[b]) for a,b in folds])
L=np.column_stack([lg(ld(m,'oof')) for m in ['lgb_r1','cat_r1','xgb_r1','lrbin_r1']])
def metaF(extra_fn):
    o=np.zeros(n)
    for a,b in folds:
        ea,eb=extra_fn(a,b); Fa=np.column_stack([L[a]]+ea); Fb=np.column_stack([L[b]]+eb)
        mu,sd=Fa.mean(0),Fa.std(0)+1e-9; o[b]=LogisticRegression(C=1.0,max_iter=2000).fit((Fa-mu)/sd,y[a]).decision_function((Fb-mu)/sd)
    return o
ref=fa(metaF(lambda a,b:([],[])))
CA=(df.AMT_CREDIT.astype(str)+'_'+df.AMT_ANNUITY.astype(str)).values
for name,keys,al in [('PRODUCT2',PRODUCT,5),('PRODUCT2',PRODUCT,10),('CREDITxANNUITY',CA,20)]:
    def f(a,b,keys=keys,alpha=al):
        ta,tb=nested_te(keys[:n],a,b,alpha); return [lg(ta)],[lg(tb)]
    d=fa(metaF(f))-ref; print(f'STRICT meta+TE {name} a={al}: {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5',flush=True)
# TE san pham: tin hieu con lai sau khi biet s1?  AUC theo nhom
s1=np.load(S+'/a2r2/s1.npy'); tev=np.zeros(n)
for a,b in folds: tev[b]=te_map(PRODUCT[:n],y,a,b,20)
cnt=pd.Series(PRODUCT).map(pd.Series(PRODUCT).value_counts()).values[:n]
for lo,hi in [(1,5),(5,30),(30,200),(200,1000),(1000,10**9)]:
    m=(cnt>=lo)&(cnt<hi); print(f'PRODUCT count [{lo},{hi}): n {m.sum()} dr {y[m].mean():.3f} AUC s1 {A(y[m],s1[m]):.4f} AUC TE {A(y[m],tev[m]):.4f}',flush=True)
# do chi tiet: TE product co tin hieu trong nhom co dinh EXT?
# (b) sang loc LGB: them TE lam dac trung (inner-fold cho train)
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); cols=[c for c in df.columns if c not in newc]+G['IMP']+G['RATE']+G['TERM']
X=df.iloc[:n][cols].reset_index(drop=True)
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
def screen(addfn,tag):
    res=[];t=time.time()
    for fo in (0,1):
        a,b=folds[fo]; Xa=X.loc[a].reset_index(drop=True); Xb=X.loc[b].reset_index(drop=True)
        for k,(va,vb) in addfn(a,b).items(): Xa[k]=va; Xb[k]=vb
        for s in (0,1):
            m=lgb.train(dict(P,seed=s),lgb.Dataset(Xa,y[a]),5000,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
            res.append(A(y[b],m.predict(Xb,num_iteration=m.best_iteration)))
    print(f'{tag:26s}',np.round(res,5),'mean',round(np.mean(res),5),f'{time.time()-t:.0f}s',flush=True); return np.array(res)
r0=screen(lambda a,b:{},'REF IMP,RATE,TERM')
def addP(a,b):
    ta,tb=nested_te(PRODUCT[:n],a,b,100); ua,ub=nested_te(TR[:n],a,b,100); return {'TE_PRODUCT':(ta,tb),'TE_TERMRATE':(ua,ub)}
r1=screen(addP,'+TE PRODUCT,TERMxRATE'); d=r1-r0; print('   diff',np.round(d,5),'mean',f'{d.mean():+.5f}','pos',(d>0).sum(),'/4')
# bien the khong dung nhan: PRODUCT la categorical high-card native? -> chi freq da co. Thu mean EXTF_MEAN theo PRODUCT (khong nhan)
def addU(a,b):
    k=pd.Series(PRODUCT); g=pd.Series(df.EXTF_MEAN.values).groupby(k).transform('mean').values; c=k.map(k.value_counts()).values
    g2=pd.Series(df.DAYS_BIRTH.values).groupby(k).transform('mean').values
    return {'PROD_EXTF_MEAN':(g[:n][a],g[:n][b]),'PROD_AGE_MEAN':(g2[:n][a],g2[:n][b])}
#r2=screen(addU,'+PRODUCT group means (no y)'); pass #,np.round(d,5),'mean',f'{d.mean():+.5f}','pos',(d>0).sum(),'/4')
# (c) uid concordance
uid=(df.CODE_GENDER.astype(str)+'_'+(df.DAYS_ID_PUBLISH-df.DAYS_BIRTH).astype(str)+'_'+(df.DAYS_REGISTRATION-df.DAYS_BIRTH).round().astype(str)).values[:n]
t=pd.DataFrame({'u':uid,'y':y}); sz=t.groupby('u').y.transform('size'); s=t.groupby('u').y.transform('sum')
oth=(s-t.y)/(sz-1).replace(0,np.nan); m=sz>1
print('uid dup rows train',int(m.sum()),' P(y=1|other y=1)',round(t.y[m&(oth>0)].mean(),3),'n',int((m&(oth>0)).sum()),' P(y=1|other all 0)',round(t.y[m&(oth==0)].mean(),3))

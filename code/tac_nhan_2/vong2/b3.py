# (1) cau truc lai suat roi rac? (2) khach hang lap lai (uid) (3) TE trong fold cua o san pham (4) meta-stack co tuong tac nhom
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from ens_final_r1 import ld, lg, folds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1(); X=df.iloc[:n].reset_index(drop=True)
s1=np.load(S+'/a2r2/s1.npy'); s0=np.load(S+'/a2r2/s0.npy')
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
# (1)
rk=np.round(df.RATE_HAT*400)  # bin 0.25%
vc=rk[cash].value_counts(normalize=True)
print('RATE_HAT bins (0.25%) among cash: n unique',vc.size,' top10 share',round(vc.head(10).sum(),3),' top30',round(vc.head(30).sum(),3))
print(vc.head(12).rename(lambda i:f'{i/4:.2f}%').round(3).to_string())
print('TERM_DENS_RATIO>2 share (cash)',round((df.TERM_DENS_RATIO[cash]>2).mean(),3),' >1.2',round((df.TERM_DENS_RATIO[cash]>1.2).mean(),3))
# AUC RATE_HAT within term
rows=[]
for t in [12,18,24,36,48,54,60]:
    m=(X.TERM_HAT==t).values
    rows.append((t,m.sum(),round(A(y[m],X.RATE_HAT[m]),4),round(A(y[m],-X.EXTF_MEAN[m]),4)))
print('AUC RATE_HAT (cao=rui ro) trong tung TERM_HAT [term,n,AUC_rate,AUC_-EXTF_MEAN]:',rows)
print('corr spearman RATE_HAT vs EXTF_MEAN (cash):',round(pd.Series(df.RATE_HAT[cash]).corr(pd.Series(df.EXTF_MEAN[cash]),method='spearman'),3))
# (2) uid: cung nguoi -> (DAYS_ID_PUBLISH-DAYS_BIRTH), (DAYS_REGISTRATION-DAYS_BIRTH) bat bien theo thoi diem nop
uid=(df.CODE_GENDER.astype(str)+'_'+(df.DAYS_ID_PUBLISH-df.DAYS_BIRTH).astype(str)+'_'+(df.DAYS_REGISTRATION-df.DAYS_BIRTH).round().astype(str))
cnt=uid.map(uid.value_counts()); print('uid(gender,ID-BIRTH,REG-BIRTH): rows in dup groups',int((cnt>1).sum()),'of',len(df),' groups',int((uid.value_counts()>1).sum()))
uid2=uid+'_'+df.EXT_SOURCE_1.round(6).astype(str); c2=uid2.map(uid2.value_counts()); print('  + EXT1 equal: dup rows',int((c2>1).sum()))
tr=pd.DataFrame({'u':uid[:n].values,'y':y}); g=tr.groupby('u').y.agg(['size','mean']); g=g[g['size']>1]
if len(g): print('  train dup groups',len(g),' rows',int(g['size'].sum()),' target rate in dup',round((g['size']*g['mean']).sum()/g['size'].sum(),3),' groups mixed y',int(((g['mean']>0)&(g['mean']<1)).sum()))
# (3) TE trong fold cho o (TERM_HAT, RATE bin 1%) va (ORG x OCC), (PRODUCT round(C/A,2)) -> them vao meta LR (long nhau)
def te(key,a,b,alpha=100):
    k=pd.Series(key); pr=y[a].mean(); d=pd.DataFrame({'k':k.iloc[a].values,'y':y[a]}).groupby('k').y.agg(['sum','count'])
    v=(d['sum']+alpha*pr)/(d['count']+alpha); return k.iloc[b].map(v).fillna(pr).values
keys={'TERMxRATE1':(X.TERM_HAT.fillna(-1).astype(str)+'_'+np.round(X.RATE_HAT*100).fillna(-1).astype(str)).values,
      'ORGxOCC':(X.ORGANIZATION_TYPE.astype(str)+'_'+X.OCCUPATION_TYPE.astype(str)).values,
      'PRODUCT':np.round(X.AMT_CREDIT/X.AMT_ANNUITY,2).astype(str).values,
      'CREDITxGOODS':(X.AMT_CREDIT.astype(str)+'_'+X.AMT_GOODS_PRICE.astype(str)).values,
      'ORGxINCxEDU':(X.ORGANIZATION_TYPE.astype(str)+'_'+X.NAME_INCOME_TYPE.astype(str)+'_'+X.NAME_EDUCATION_TYPE.astype(str)).values}
def fa(o): return np.array([A(y[b],o[b]) for a,b in folds])
def metaLR(F):
    o=np.zeros(n)
    for a,b in folds:
        mu,sd=F[a].mean(0),F[a].std(0)+1e-9; o[b]=LogisticRegression(C=1.0,max_iter=2000).fit((F[a]-mu)/sd,y[a]).decision_function((F[b]-mu)/sd)
    return o
L=np.column_stack([lg(ld(m,'oof')) for m in ['lgb_r1','cat_r1','xgb_r1','lrbin_r1']])
ref=fa(metaLR(L)); print('meta ref',np.round(ref,5),round(A(y,metaLR(L)),5))
for k,v in keys.items():
    # TE OOF: cho meta can TE tinh trong fold ngoai (a) -> b; trong a dung inner 5-fold de tranh lech
    tev=np.zeros(n)
    for a,b in folds: tev[b]=te(v,a,b)
    print(f'TE {k}: n_levels {len(set(v))}, AUC don {A(y,tev):.4f}, spearman vs s1 {pd.Series(tev).corr(pd.Series(s1),method="spearman"):.3f}',flush=True)
    # meta: dung tev (OOF-of-fold) cho ca a va b (moi gia tri cua 1 dong deu tinh khong co nhan cua chinh no)
    o=metaLR(np.column_stack([L,lg(tev)])); d=fa(o)-ref; print(f'   meta+TE {k}: d {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5')
# (4) meta co tuong tac nhom
ind={'EXT_NA2':(X.EXT_NA==2).values,'PENS':(X.DAYS_EMPLOYED_ANOM==1).values,'REV':(X.NAME_CONTRACT_TYPE!='Cash loans').values,'AGE60':(X.DAYS_BIRTH<-60*365.25).values,'MALE':(X.CODE_GENDER=='M').values}
Ig=np.column_stack(list(ind.values())).astype(float)
F=np.column_stack([L,Ig]+[L*Ig[:,[j]] for j in range(Ig.shape[1])]); o=metaLR(F); d=fa(o)-ref
print('meta LR + nhom + logit*nhom:',np.round(d,5),'mean',f'{d.mean():+.5f}',(d>0).sum(),'/5')
# LGB meta nong voi vai bien goc
import lightgbm as lgb
R=np.column_stack([L,X[['EXT_NA','DAYS_BIRTH','DAYS_EMPLOYED_ANOM','AMT_CREDIT','CREDIT_ANNUITY','TERM_HAT']].values.astype(float)])
o=np.zeros(n)
for a,b in folds:
    m=lgb.train(dict(objective='binary',learning_rate=0.02,num_leaves=4,min_child_samples=500,lambda_l2=10,verbose=-1,n_jobs=4,seed=0),lgb.Dataset(R[a],y[a]),400)
    o[b]=m.predict(R[b])
d=fa(o)-ref; print('meta LGB (4 la,400 vong) logit+6 bien goc:',np.round(d,5),'mean',f'{d.mean():+.5f}',(d>0).sum(),'/5')

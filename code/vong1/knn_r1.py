# KNN target-mean LONG trong fold (khong ro nhan).
# Voi moi outer fold k (5-fold rs=42): dong train Tk -> dac trung tinh bang inner 5-fold tren Tk;
# dong valid fold k va test -> lang gieng trong toan bo Tk. Luu r1_knn.pkl: list 5 dict {tr, va, te} (DataFrame).
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
KS=[50,200,500]; KMAX=max(KS)
def space(df):
    Z1=df[['EXT_SOURCE_1_FILL','EXT_SOURCE_2_FILL','EXT_SOURCE_3_FILL','CREDIT_ANNUITY']].copy()
    Z2=df[['EXTF_MEAN','EXT_SOURCE_2_FILL','EXT_SOURCE_3_FILL','CREDIT_ANNUITY','DAYS_BIRTH']].copy()
    out=[]
    for Z in (Z1,Z2):
        Z=Z.fillna(Z.median())
        out.append(QuantileTransformer(n_quantiles=1000,output_distribution='normal',subsample=10**9,random_state=0).fit_transform(Z.values))
    return out
def feats(Z,yfit,ifit,iq,tag):
    nn=NearestNeighbors(n_neighbors=KMAX,algorithm='kd_tree',n_jobs=4).fit(Z[ifit])
    _,ind=nn.kneighbors(Z[iq]); yy=yfit[ind]
    return {f'KNN{tag}_{k}':yy[:,:k].mean(1) for k in KS}
def build(df,y,n):
    Zs=space(df); te=np.arange(n,len(df))
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y)); res=[]
    for f,(a,b) in enumerate(folds):
        t=time.time(); d={'tr':{},'va':{},'te':{}}
        for zi,Z in enumerate(Zs):
            tag=str(zi+1); trf={f'KNN{tag}_{k}':np.zeros(len(a)) for k in KS}
            for ia,ib in StratifiedKFold(5,shuffle=True,random_state=f).split(a,y[a]):
                r=feats(Z,y[a][ia],a[ia],a[ib],tag)
                for k in r: trf[k][ib]=r[k]
            d['tr'].update(trf)
            d['va'].update(feats(Z,y[a],a,b,tag)); d['te'].update(feats(Z,y[a],a,te,tag))
        res.append({k:pd.DataFrame(v) for k,v in d.items()}); print('knn fold',f,f'{time.time()-t:.0f}s',flush=True)
    return res
if __name__=='__main__':
    df,y,n,cats,ids,G=load_r1()
    res=build(df,y,n); pd.to_pickle(res,S+'/r1_knn.pkl')
    from sklearn.metrics import roc_auc_score as A
    for f in range(5):
        print(f,{k:round(A(y[list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))[f][1]],v),4) for k,v in res[f]['va'].items()})

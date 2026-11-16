# Vong 2: fe_r1 + "ho so quan the cua san pham vay" (target-free, train+test) + cac nhom thu nghiem (RATEG, IMPB).
# load_r2() tra ve (df, y, n, cats, ids, G); G them cac nhom:
#   PROF  : khoa TERM_HAT x round(RATE_HAT,0.5%) -> mean {EXTF_MEAN, EXT2, EXT3, EXT1_PRED, tuoi, tham nien, ti le nam} + std EXTF_MEAN (8 cot)
#   PROFX : cung khoa, them {log thu nhap, log tien vay, ti le bao hiem, ti le thieu EXT3, ti le hoc van cao, so dong} (6 cot)
#   PROFR : 5 khoa round(RATE_n,0.5%) n=12/24/36/48/60 x {EXTF_MEAN, EXT3, tuoi, tham nien} (20 cot)
#   PROFC : khoa round(C/A,2) x {EXTF_MEAN, EXT3, tuoi, tham nien} (4 cot)
#   RATEG : lai suat ngam dinh voi AMT_GOODS_PRICE lam goc (L1.1c) n=12/24/36/48/60 (5 cot)
#   IMPR  : EXT_k du doan bang thuat toan thu hai (Ridge tren bin one-hot) (3 cot);  IMPB: bo IMP dung trung binh LGB+Ridge (15 cot, hau to _B)
# KHONG cot nao dung TARGET. Khoa san pham (dung cho TE trong fold o cho khac): PRODUCT_KEY(df).
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,S)
from fe_r1 import load_r1, rate, E
CACHE2=S+'/r2_imp_ridge.pkl'

def profile(key,cols,pre,minc=30,std=True):
    p=pd.Series(key).fillna(-9).values; d=cols.copy(); d['p']=p; s=pd.Series(p); cnt=s.map(s.value_counts()).values
    g=d.groupby('p'); o=pd.DataFrame({f'{pre}_{c}':g[c].transform('mean').values for c in cols.columns})
    if std: o[pre+'_estd']=g['e'].transform('std').values
    o[cnt<minc]=np.nan; return o,cnt

def ridge_impute(df,cats):
    """Thuat toan dien khuyet thu hai: Ridge tren one-hot bin quantile (16 bin) cua cac cot khong chua EXT; OOF KFold 5 nhu fe_r1."""
    import scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    EXTD=[c for c in df.columns if 'EXT' in c or c.startswith('E1') or c.startswith('E2')]
    num=[c for c in df.columns if c not in EXTD and c not in cats]
    def binblk(x,NB=16):
        ok=~np.isnan(x)
        if np.unique(x[ok]).size<=NB: codes=pd.factorize(np.where(ok,x,-9e9))[0]
        else:
            qs=np.unique(np.nanquantile(x,np.linspace(0,1,NB+1)[1:-1])); codes=np.searchsorted(qs,x); codes[~ok]=len(qs)+1
        return sp.csr_matrix(pd.get_dummies(codes.astype(str)).astype(np.float32).values)
    base=[binblk(df[c].values.astype(float)) for c in num]+[sp.csr_matrix(pd.get_dummies(df[c].astype(str)).astype(np.float32).values) for c in cats]
    eb={e:binblk(df[e].values.astype(float)) for e in E}; out={}
    for tgt in E:
        M=sp.hstack(base+[eb[e] for e in E if e!=tgt]).tocsr(); have=df[tgt].notna().values; hv=np.where(have)[0]; nh=np.where(~have)[0]; pred=np.zeros(len(df))
        for a,b in KFold(5,shuffle=True,random_state=0).split(hv):
            m=Ridge(alpha=20.0,solver='sag',max_iter=200,random_state=0).fit(M[hv[a]],df[tgt].values[hv[a]])
            pred[hv[b]]=m.predict(M[hv[b]]); pred[nh]+=m.predict(M[nh])/5
        print(tgt,'ridge imputer corr',round(np.corrcoef(pred[hv],df[tgt].values[hv])[0,1],3),flush=True); out[tgt+'_PREDR']=pred
    return pd.DataFrame(out)

def load_r2(extra=True):
    df,y,n,cats,ids,g=load_r1()
    C=df.AMT_CREDIT.values.astype(float); A=df.AMT_ANNUITY.values.astype(float); Gd=df.AMT_GOODS_PRICE.values.astype(float)
    cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
    core=pd.DataFrame({'e':df.EXTF_MEAN.values,'e2':df.EXT_SOURCE_2.values,'e3':df.EXT_SOURCE_3.values,'e1p':df.EXT_SOURCE_1_PRED.values,
                       'age':df.DAYS_BIRTH.values,'emp':df.DAYS_EMPLOYED.values,'male':(df.CODE_GENDER=='M').values.astype(float)})
    new={}
    kTR=np.where(cash,df.TERM_HAT.values*100000+np.round(df.RATE_HAT.values*200),-5)
    p,cnt=profile(kTR,core,'PT'); g['PROF']=list(p.columns)
    for c in p.columns: new[c]=p[c].values
    if extra:
        xt=pd.DataFrame({'e':core.e,'inc':np.log(df.AMT_INCOME_TOTAL.values),'cr':np.log(C),'ins':(C>Gd+1).astype(float),
                         'e3na':df.EXT_SOURCE_3.isna().values.astype(float),'hiedu':df.NAME_EDUCATION_TYPE.astype(str).str.contains('Higher').values.astype(float)})
        p,_=profile(kTR,xt,'PT',std=False); p=p.drop(columns='PT_e'); p['PT_cnt']=np.where(cnt<30,np.nan,cnt)
        g['PROFX']=list(p.columns)
        for c in p.columns: new[c]=p[c].values
        g['PROFR']=[]; g['RATEG']=[]
        for nn in (12,24,36,48,60):
            r=rate(C,A,nn)*12
            p,_=profile(np.where(cash&~np.isnan(r),np.round(r*200),-5),core[['e','e3','age','emp']],f'PR{nn}',std=False)
            for c in p.columns: new[c]=p[c].values; g['PROFR'].append(c)
            rg=rate(Gd,A,nn)*12; rg[~cash]=np.nan; new[f'RATEG_{nn}']=rg; g['RATEG'].append(f'RATEG_{nn}')
        p,_=profile(np.where(cash,np.round(C/A,2),-5.0),core[['e','e3','age','emp']],'PC',std=False); g['PROFC']=list(p.columns)
        for c in p.columns: new[c]=p[c].values
        # ---- thuat toan dien khuyet thu hai (L1.1c)
        if os.path.exists(CACHE2): ir=pd.read_pickle(CACHE2)
        else: ir=ridge_impute(df[[c for c in df.columns if c not in sum(g.values(),[])]],cats); ir.to_pickle(CACHE2)
        g['IMPR']=list(ir.columns)
        for c in ir.columns: new[c]=ir[c].values
        nb={}
        for e in E:
            pb=(df[e+'_PRED'].values+ir[e+'_PREDR'].values)/2; nb[e+'_PRED_B']=pb; nb[e+'_FILL_B']=df[e].fillna(pd.Series(pb)).values; nb[e+'_RES_B']=(df[e]-pb).values
        F=pd.DataFrame({e:nb[e+'_FILL_B'] for e in E})
        nb['EXTF_MEAN_B']=F.mean(1).values; nb['EXTF_PROD_B']=F.prod(1).values; nb['EXTF_MIN_B']=F.min(1).values; nb['EXTF_MAX_B']=F.max(1).values
        nb['EXTF_STD_B']=F.std(1).values; nb['EXTF_W_B']=(F.EXT_SOURCE_1*2+F.EXT_SOURCE_2+F.EXT_SOURCE_3*3).values
        g['IMPB']=list(nb); new.update(nb)
    df=pd.concat([df,pd.DataFrame(new)],axis=1).replace([np.inf,-np.inf],np.nan)
    return df,y,n,cats,ids,g

def PRODUCT_KEY(df): return np.round(df.AMT_CREDIT/df.AMT_ANNUITY,2).astype(str).values

def te_map(keys,yy,idx_fit,idx_apply,alpha=10):
    """TE lam min (sum+alpha*prior)/(count+alpha): hoc tren idx_fit (co nhan), ap cho idx_apply (chi so trong keys)."""
    pr=yy[idx_fit].mean(); d=pd.DataFrame({'k':keys[idx_fit],'y':yy[idx_fit]}).groupby('k').y.agg(['sum','count'])
    v=(d['sum']+alpha*pr)/(d['count']+alpha); return pd.Series(keys[idx_apply]).map(v).fillna(pr).values

def nested_te(keys,yy,a,apply_list,alpha=10,seed=0):
    """R14: TE cho phan train a bang inner 5-fold TRONG a; cho moi tap trong apply_list (valid, test) tu toan bo a. Nhan ngoai a khong bao gio duoc dung."""
    from sklearn.model_selection import StratifiedKFold
    ta=np.zeros(len(a))
    for ia,ib in StratifiedKFold(5,shuffle=True,random_state=seed).split(a,yy[a]): ta[ib]=te_map(keys,yy,a[ia],a[ib],alpha)
    return ta,[te_map(keys,yy,a,idx,alpha) for idx in apply_list]

if __name__=='__main__':
    df,y,n,cats,ids,g=load_r2(); print(df.shape,{k:len(v) for k,v in g.items()})

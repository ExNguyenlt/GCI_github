# Vong 3: fe_r2 (r2_df.pkl) + nhom PQ (L3.3) + ban dung lai ky han co phat lai suat cao (L2.3b). KHONG cot nao dung TARGET.
#   load_r3() -> (df, y, n, cats, ids, G)  ; G them 'PQ' (4 cot: ho so khong nhan train+test theo khoa TERM x RATE(0.5%) x ngu phan vi AMT_CREDIT; < 30 dong -> NaN)
#   term_alt(df, mode) -> DataFrame cac cot TERM (6 cot phu thuoc ky han) + PROF (8 cot) + PQ (4 cot) dung lai voi tien nghiem lai suat:
#       mode='soft35': mat do * exp(-(r-0.35)/0.08) khi r > 35%/nam ;  mode='hard40': loai ky han co r > 40%/nam (ban goc: chi loai r > 60%)
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,S)
from fe_r1 import rate
from fe_r2 import profile, PRODUCT_KEY, nested_te, te_map
PK2=S+'/r2_df.pkl'; PK3=S+'/r3_df.pkl'
TL=[6,10,12,18,24,30,36,42,48,54,60]

def _core(df,ext='e'):
    return pd.DataFrame({'e':df.EXTF_MEAN.values,'e2':df.EXT_SOURCE_2.values,'e3':df.EXT_SOURCE_3.values,'e1p':df.EXT_SOURCE_1_PRED.values,
                         'age':df.DAYS_BIRTH.values,'emp':df.DAYS_EMPLOYED.values,'male':(df.CODE_GENDER=='M').values.astype(float)})
def _ktr(df,th,rh):
    cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
    return pd.Series(np.where(cash,th*100000+np.round(rh*200),-5)).fillna(-9).values
def pq_cols(df,th,rh):
    core=_core(df)[['e','e3','age','emp']]
    p,_=profile(_ktr(df,th,rh)*10+pd.qcut(df.AMT_CREDIT,5,labels=False).values,core,'PQ',std=False); return p

def term_alt(df,mode):
    C=df.AMT_CREDIT.values.astype(float); A=df.AMT_ANNUITY.values.astype(float); rev=(df.NAME_CONTRACT_TYPE!='Cash loans').values
    rates={}; dens=[]
    for nn in TL:
        r=rate(C,A,nn)*12; r[rev]=np.nan; rates[nn]=r
        key=pd.Series(np.round(r*400)); cnt=key.map(key.value_counts()).fillna(0).values.astype(float); cnt[np.isnan(r)]=0; cnt[r>0.6]=0
        if mode=='soft35': cnt=cnt*np.where(r>0.35,np.exp(-(np.nan_to_num(r)-0.35)/0.08),1.0)
        elif mode=='hard40': cnt[r>0.40]=0
        dens.append(cnt)
    D=np.vstack(dens).T; j=D.argmax(1); srt=np.sort(D,1)
    th=np.array(TL,float)[j]; rh=np.vstack([rates[t] for t in TL]).T[np.arange(len(j)),j]
    th[rev|(D.max(1)==0)]=np.nan; rh[np.isnan(th)]=np.nan
    o={'TERM_HAT':th,'RATE_HAT':rh,'TERM_DENS':np.where(np.isnan(th),np.nan,srt[:,-1]),'TERM_DENS_RATIO':np.where(np.isnan(th),np.nan,srt[:,-1]/(srt[:,-2]+1)),
       'OVERPAY_REL':A*th/C-1,'AGE_END_LOAN':-df.DAYS_BIRTH.values/365.25+th/12}
    p,_=profile(_ktr(df,th,rh),_core(df),'PT')
    out=pd.concat([pd.DataFrame(o),p,pq_cols(df,th,rh)],axis=1).replace([np.inf,-np.inf],np.nan)
    return out

def load_r3():
    if os.path.exists(PK3): return pd.read_pickle(PK3)
    df,y,n,cats,ids,G=pd.read_pickle(PK2)
    p=pq_cols(df,df.TERM_HAT.values,df.RATE_HAT.values)
    for c in p.columns: df[c]=p[c].values
    G['PQ']=list(p.columns); G['RES']=[e+'_RES' for e in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']]
    r=(df,y,n,cats,ids,G); pd.to_pickle(r,PK3); return r

if __name__=='__main__':
    df,y,n,cats,ids,G=load_r3(); print(df.shape,{k:len(v) for k,v in G.items()})
    for mode in ('orig','soft35','hard40'):
        t=term_alt(df,mode); ch=(t.TERM_HAT.fillna(-1).values!=df.TERM_HAT.fillna(-1).values)
        print(mode,'TERM_HAT doi o %d dong (%.2f%%)'%(ch.sum(),100*ch.mean()),'| RATE_HAT>35%%: %.3f -> %.3f'%((df.RATE_HAT>0.35).mean(),(t.RATE_HAT>0.35).mean()),
              '| max abs diff PROF',float(np.nanmax(np.abs(t.PT_e.values-df.PT_e.values))) if mode=='orig' else '')

# Vong 1: fe.py + dien khuyet EXT bang mo hinh (khong dung TARGET) + lai suat ngam dinh / dung lai ky han + dac trung ngay/tuoi.
# load_r1() tra ve (df, y, n, cats, ids, groups) ; groups = dict ten nhom -> danh sach cot moi (de sang loc).
import os, sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,S)
from fe import load
from sklearn.model_selection import KFold
CACHE=S+'/r1_feats.pkl'
E=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

def rate(C,A,nn):
    """lai thang r giai A = C*r/(1-(1+r)^-n) (chia doi); NaN neu A*n<=C"""
    lo=np.full_like(C,1e-7); hi=np.full_like(C,0.25)
    for _ in range(60):
        mid=(lo+hi)/2; pv=A*(1-(1+mid)**-nn)/mid
        lo=np.where(pv>C,mid,lo); hi=np.where(pv>C,hi,mid)
    r=(lo+hi)/2; r[~(A*nn>C)]=np.nan; return r

def impute(df,cats):
    d=df.copy()
    for c in cats: d[c]=d[c].astype('category')
    EXTD=[c for c in d.columns if 'EXT' in c or c.startswith('E1') or c.startswith('E2')]
    base=[c for c in d.columns if c not in EXTD]
    out={}
    Pr=dict(objective='regression',learning_rate=0.05,num_leaves=31,feature_fraction=0.5,min_child_samples=50,verbose=-1,n_jobs=4)
    for tgt in E:
        cols=base+[e for e in E if e!=tgt]; have=d[tgt].notna().values; pred=np.zeros(len(d))
        hv=np.where(have)[0]; nh=np.where(~have)[0]
        for a,b in KFold(5,shuffle=True,random_state=0).split(hv):
            m=lgb.train(Pr,lgb.Dataset(d.iloc[hv[a]][cols],d[tgt].values[hv[a]]),800)
            pred[hv[b]]=m.predict(d.iloc[hv[b]][cols]); pred[nh]+=m.predict(d.iloc[nh][cols])/5
        r=np.corrcoef(pred[hv],d[tgt].values[hv])[0,1]; print(tgt,'imputer corr',round(r,3),flush=True)
        out[tgt+'_PRED']=pred
    return pd.DataFrame(out)

def load_r1(verbose=True):
    df,y,n,cats,ids=load()
    if os.path.exists(CACHE): imp=pd.read_pickle(CACHE)
    else: imp=impute(df,cats); imp.to_pickle(CACHE)
    g={}
    # ---- IMP: EXT dien khuyet + residual + tong hop tren ban dien
    new={}
    for e in E:
        p=imp[e+'_PRED'].values; new[e+'_PRED']=p; new[e+'_FILL']=df[e].fillna(pd.Series(p)).values; new[e+'_RES']=(df[e]-p).values
    F=pd.DataFrame({e:new[e+'_FILL'] for e in E})
    new['EXTF_MEAN']=F.mean(1).values; new['EXTF_PROD']=F.prod(1).values; new['EXTF_MIN']=F.min(1).values; new['EXTF_MAX']=F.max(1).values
    new['EXTF_STD']=F.std(1).values; new['EXTF_W']=(F.EXT_SOURCE_1*2+F.EXT_SOURCE_2+F.EXT_SOURCE_3*3).values
    g['IMP']=list(new)
    # ---- RATE: lai suat ngam dinh cho cac ky han co dinh (Revolving -> NaN)
    C=df.AMT_CREDIT.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
    rev=(df.NAME_CONTRACT_TYPE!='Cash loans').values
    rates={}
    for nn in [6,10,12,18,24,30,36,42,48,54,60]:
        r=rate(C,A,nn)*12; r[rev]=np.nan; rates[nn]=r
    rc=[]
    for nn in [12,24,36,48,60]: new[f'RATE_{nn}']=rates[nn]; rc.append(f'RATE_{nn}')
    g['RATE']=rc
    # ---- TERM: dung lai ky han = ky han co mat do lai suat cao nhat (train+test, khong TARGET)
    T=[6,10,12,18,24,30,36,42,48,54,60]; dens=[]
    for nn in T:
        key=pd.Series(np.round(rates[nn]*400)); cnt=key.map(key.value_counts()).fillna(0).values.astype(float)
        cnt[np.isnan(rates[nn])]=0; cnt[(rates[nn]>0.6)]=0; dens.append(cnt)
    D=np.vstack(dens).T; j=D.argmax(1); srt=np.sort(D,1)
    th=np.array(T,float)[j]; rh=np.vstack([rates[t] for t in T]).T[np.arange(len(j)),j]
    th[rev|(D.max(1)==0)]=np.nan; rh[np.isnan(th)]=np.nan
    new['TERM_HAT']=th; new['RATE_HAT']=rh; new['TERM_DENS']=np.where(np.isnan(th),np.nan,srt[:,-1])
    new['TERM_DENS_RATIO']=np.where(np.isnan(th),np.nan,srt[:,-1]/(srt[:,-2]+1))
    new['OVERPAY_REL']=A*th/C-1; new['AGE_END_LOAN']=-df.DAYS_BIRTH.values/365.25+th/12
    prod=pd.Series(np.round(C/A,2)); new['PRODUCT_FREQ']=prod.map(prod.value_counts()).values.astype(float)
    pg=pd.Series(np.round(C/df.AMT_GOODS_PRICE.values,3)); new['CG_FREQ']=pg.map(pg.value_counts()).values.astype(float)
    g['TERM']=['TERM_HAT','RATE_HAT','TERM_DENS','TERM_DENS_RATIO','OVERPAY_REL','AGE_END_LOAN','PRODUCT_FREQ','CG_FREQ']
    # ---- DATE: tuoi tai cac moc / tre doi ho chieu (moc 14/20/45 tuoi)
    age=-df.DAYS_BIRTH/365.25
    for c in ['DAYS_ID_PUBLISH','DAYS_REGISTRATION','DAYS_EMPLOYED','DAYS_LAST_PHONE_CHANGE']:
        new[c+'_AGE_AT']=((df[c]-df.DAYS_BIRTH)/365.25).values
    ida=pd.Series(new['DAYS_ID_PUBLISH_AGE_AT'])
    new['ID_LATE']=np.where(ida>=45,ida-45,np.where(ida>=20,ida-20,ida-14))
    new['ID_OVERDUE']=(((age>=45)&(ida<45))|((age>=20)&(ida<20))).astype(int).values
    new['ID_OVERDUE_YRS']=np.where(age>=45,np.where(ida<45,age-45,0),np.where(age>=20,np.where(ida<20,age-20,0),0))
    new['AGE_FRAC']=(age%1).values; new['YEARS_TO_45']=(45-age).values
    new['ID_MINUS_EMP']=(df.DAYS_ID_PUBLISH-df.DAYS_EMPLOYED).values; new['ID_MINUS_REG']=(df.DAYS_ID_PUBLISH-df.DAYS_REGISTRATION).values
    new['GOODS_ANNUITY']=(df.AMT_GOODS_PRICE/df.AMT_ANNUITY).values
    g['DATE']=['DAYS_ID_PUBLISH_AGE_AT','DAYS_REGISTRATION_AGE_AT','DAYS_EMPLOYED_AGE_AT','DAYS_LAST_PHONE_CHANGE_AGE_AT','ID_LATE','ID_OVERDUE',
               'ID_OVERDUE_YRS','AGE_FRAC','YEARS_TO_45','ID_MINUS_EMP','ID_MINUS_REG','GOODS_ANNUITY']
    df=pd.concat([df,pd.DataFrame(new)],axis=1).replace([np.inf,-np.inf],np.nan)
    return df,y,n,cats,ids,g

if __name__=='__main__':
    df,y,n,cats,ids,g=load_r1(); print(df.shape,{k:len(v) for k,v in g.items()})
    from sklearn.metrics import roc_auc_score as AUC
    for k,v in g.items():
        for c in v:
            x=df[c].values[:n]; ok=~np.isnan(x)
            if ok.sum()>1000: a=AUC(y[ok],x[ok]); print(k,c,round(max(a,1-a),4),'na',round(1-ok.mean(),3))

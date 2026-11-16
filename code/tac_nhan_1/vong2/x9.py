# voi tung goi bao hiem k: ky han nao cho phan phoi lai suat "roi rac" nhat (HHI cua bin 0.05%/nam)
import sys,numpy as np,pandas as pd
S=sys.argv[1]; sys.path.insert(0,S)
from fe_r1 import rate
df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl'); y=np.load(S+'/loop/a1r2/y.npy'); n=len(y)
C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
k=(C/G-1)/0.0132; kr=np.round(k); ok=np.abs(k-kr)<0.02
T=list(range(6,85,6))+[10]
R={t:rate(C,A,t)*12 for t in T}
yy=np.r_[y,np.full(len(df)-n,np.nan)]
for g in [0,3,4,5,6,8,9,10,11,12,15,16,18,20,24,30,32,40]:
    m=cash&ok&(kr==g) if g>0 else cash&np.isclose(C,G)
    res=[]
    for t in sorted(T):
        r=R[t][m]; v=~np.isnan(r)&(r<0.8)
        if v.mean()<0.5: continue
        s=pd.Series(np.round(r[v]*2000)).value_counts(normalize=True)
        res.append((t,round(v.mean(),2),round((s**2).sum()*v.mean(),4),round(s.head(5).sum()*v.mean(),3)))
    res.sort(key=lambda z:-z[2])
    print('k',g,'n',m.sum(),'dr',round(np.nanmean(yy[m]),4),'best (term,valid,HHI,top5):',res[:4])

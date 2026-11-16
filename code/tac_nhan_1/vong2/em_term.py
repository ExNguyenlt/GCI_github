# Ky han an bang EM (mo hinh hon hop tiem an), co dieu kien theo "goi bao hiem" k=(C/G-1)/0.0132 ; KHONG dung TARGET
import sys,numpy as np,pandas as pd
def rate(C,A,nn):
    lo=np.full_like(C,1e-7); hi=np.full_like(C,0.25)
    for _ in range(60):
        mid=(lo+hi)/2; pv=A*(1-(1+mid)**-nn)/mid
        lo=np.where(pv>C,mid,lo); hi=np.where(pv>C,hi,mid)
    r=(lo+hi)/2; r[~(A*nn>C)]=np.nan; return r
def em_term(df,T=(6,10,12,18,24,30,36,42,48,54,60),res=400,cond=True,iters=30,alpha=0.5,verbose=True):
    C=df.AMT_CREDIT.values.astype(float); G=df.AMT_GOODS_PRICE.values.astype(float); A=df.AMT_ANNUITY.values.astype(float)
    cash=(df.NAME_CONTRACT_TYPE=='Cash loans').values
    k=(C/G-1)/0.0132; kr=np.round(k); okk=np.abs(k-kr)<0.02
    grp=np.where(np.isclose(C,G),0,np.where(okk,kr,-1)).astype(int)   # -1 = bao hiem khong theo luoi
    if not cond: grp[:]=0
    R=np.vstack([rate(C,A,t)*12 for t in T]).T              # N x nT
    valid=(~np.isnan(R))&(R>0.0)&(R<0.6)&cash[:,None]
    B=np.where(valid,np.round(np.nan_to_num(R)*res),-1).astype(int); nb=int(0.6*res)+2
    W=valid/np.maximum(valid.sum(1,keepdims=True),1)          # khoi tao deu
    gs=np.unique(grp)
    for it in range(iters):
        L=np.zeros_like(W)
        for g in gs:
            m=grp==g
            for j in range(len(T)):
                v=m&valid[:,j]; h=np.bincount(B[v,j],weights=W[v,j],minlength=nb)+alpha/nb
                th=h/ (W[m].sum()+alpha*len(T))
                L[v,j]=th[B[v,j]]
        Wn=L/np.maximum(L.sum(1,keepdims=True),1e-300); Wn[~valid]=0
        ch=np.abs(Wn-W).max(); W=Wn
        if verbose and it%5==0: print('iter',it,'max change',round(ch,4),flush=True)
    has=valid.any(1); j=W.argmax(1); Ta=np.array(T,float)
    out=pd.DataFrame(index=df.index)
    out['TERM_EM']=np.where(has,Ta[j],np.nan); out['RATE_EM']=np.where(has,R[np.arange(len(j)),j],np.nan)
    out['POST_MAX']=np.where(has,W.max(1),np.nan)
    out['TERM_EXP']=np.where(has,(W*Ta).sum(1),np.nan); out['RATE_EXP']=np.where(has,(W*np.nan_to_num(R)).sum(1),np.nan)
    out['TERM_ENT']=np.where(has,-(W*np.log(W+1e-12)).sum(1),np.nan)
    out['INS_K']=np.where(grp>0,grp,np.where(grp==0,0,np.nan)).astype(float)
    out['INS_PER_M']=out.INS_K*0.0132/out.TERM_EM
    return out
if __name__=='__main__':
    S=sys.argv[1]; df=pd.read_pickle(S+'/loop/a1r2/df_r1.pkl')
    for cond in [False,True]:
        o=em_term(df,cond=cond); o.to_pickle(S+f'/loop/a1r2/em_cond{int(cond)}.pkl')
        print('cond',cond,'POST_MAX quantiles',np.nanquantile(o.POST_MAX,[.1,.25,.5,.75]).round(3))
        print(pd.crosstab(o.TERM_EM,df.TERM_HAT).to_string())

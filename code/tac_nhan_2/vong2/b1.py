# Phan tich nhom, tuong quan, loi lon nhat (khong huan luyen)
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from ens_final_r1 import stack, rankavg, fa, ld, lg
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1(); X=df.iloc[:n].reset_index(drop=True)
R0=['lgb','cat','xgb','lgb_opt','cat_opt','xgb_opt']; G1=['lgb_r1','cat_r1','xgb_r1']
s0=stack(R0)[0]; s1=stack(G1+['lrbin_r1'])[0]
np.save(S+'/a2r2/s0.npy',s0); np.save(S+'/a2r2/s1.npy',s1)
print('S0',round(A(y,s0),5),'S1',round(A(y,s1),5))
M={m:ld(m,'oof') for m in G1+['lrbin_r1','mlp_r1','lgb','cat']}
print('--- spearman OOF r1')
C=pd.DataFrame({k:rankdata(v) for k,v in M.items()}).corr(); print(C.round(4).to_string())
age=-X.DAYS_BIRTH/365.25
grp={
 'EXT_NA':X.EXT_NA.astype(int).astype(str),
 'EXT1_NA':X.EXT_SOURCE_1.isna().map({True:'na',False:'have'}),
 'EXT3_NA':X.EXT_SOURCE_3.isna().map({True:'na',False:'have'}),
 'CONTRACT':X.NAME_CONTRACT_TYPE.astype(str),
 'PENSION':X.DAYS_EMPLOYED_ANOM.astype(str),
 'AGE':pd.cut(age,[0,30,40,50,60,100]).astype(str),
 'GENDER':X.CODE_GENDER.astype(str),
 'INC_TYPE':X.NAME_INCOME_TYPE.astype(str),
 'TERM_HAT':X.TERM_HAT.fillna(-1).astype(int).astype(str),
 'TDR':pd.qcut(X.TERM_DENS_RATIO,4,duplicates='drop').astype(str).replace('nan','NA'),
 'EDU':X.NAME_EDUCATION_TYPE.astype(str),
 'GOODS_NA':X.AMT_GOODS_PRICE.isna().astype(str),
}
rows=[]
for g,v in grp.items():
    v=v.astype(object).where(v.notna(),"NA").astype(str)
    for lv in sorted(v.unique()):
        m=(v==lv).values
        if m.sum()<1500 or y[m].sum()<60: continue
        a0=A(y[m],s0[m]); a1=A(y[m],s1[m]); al=A(y[m],M['lgb_r1'][m]); ab=A(y[m],M['lrbin_r1'][m]); am=A(y[m],M['mlp_r1'][m])
        rows.append((g,lv,m.sum(),round(m.mean(),3),round(y[m].mean(),3),round(a0,4),round(a1,4),round(a1-a0,4),round(al,4),round(ab,4),round(am,4),round(ab-al,4)))
T=pd.DataFrame(rows,columns=['grp','lvl','n','share','dr','AUC_s0','AUC_s1','d','lgb_r1','lrbin','mlp','lrbin-lgb'])
pd.set_option('display.width',250); print(T.to_string())
# do loi lon nhat
r=rankdata(s1)/n
fp=(y==0)&(r>0.97); fn=(y==1)&(r<0.25)
print('FP (y=0, top3%)',fp.sum(),' FN (y=1, rank<25%)',fn.sum(), 'share of positives in low25', round(fn.sum()/y.sum(),3))
cols=['EXT_NA','EXTF_MEAN','DAYS_EMPLOYED_ANOM','CREDIT_ANNUITY','AMT_CREDIT','AMT_INCOME_TOTAL','RATE_HAT','TERM_HAT','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
Z=X[cols].copy(); Z['AGE']=age; Z['MALE']=(X.CODE_GENDER=='M').astype(int); Z['REV']=(X.NAME_CONTRACT_TYPE!='Cash loans').astype(int)
Z['LOW_EDU']=X.NAME_EDUCATION_TYPE.isin(['Secondary / secondary special','Lower secondary']).astype(int)
Z['E1NA']=X.EXT_SOURCE_1.isna().astype(int); Z['E3NA']=X.EXT_SOURCE_3.isna().astype(int)
pos=y==1; neg=y==0
print(pd.DataFrame({'all':Z.median(numeric_only=True),'pos_all':Z[pos].median(),'FN_pos_lowrank':Z[fn].median(),'neg_all':Z[neg].median(),'FP_neg_top':Z[fp].median()}).round(3).to_string())
print('means of binary:')
print(pd.DataFrame({'all':Z.mean(),'pos_all':Z[pos].mean(),'FN':Z[fn].mean(),'neg':Z[neg].mean(),'FP':Z[fp].mean()}).loc[['MALE','REV','LOW_EDU','E1NA','E3NA','DAYS_EMPLOYED_ANOM']].round(3).to_string())
# FN: nhung nguoi vo no voi EXT cao
print('FN EXTF_MEAN quantiles',np.round(np.quantile(X.EXTF_MEAN[fn],[.1,.5,.9]),3),' all pos',np.round(np.quantile(X.EXTF_MEAN[pos],[.1,.5,.9]),3))
# ty le loi: phan dong gop "between-group" vs within

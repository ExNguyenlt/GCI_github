# L2.8 / R20: bang AUC theo nhom cho stack moi (oof_final_r2.npy) so voi stack vong 1 (a2r2/s1.npy)
import sys, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=pd.read_pickle(S+'/r2_df.pkl'); X=df.iloc[:n].reset_index(drop=True)
s1=np.load(S+'/a2r2/s1.npy'); s2=np.load(S+'/oof_final_r2.npy'); age=-X.DAYS_BIRTH/365.25
grp={'ALL':pd.Series(['all']*n),'EXT_NA':X.EXT_NA.astype(int).astype(str),'EXT3_NA':X.EXT_SOURCE_3.isna().map({True:'na',False:'have'}),
 'CONTRACT':X.NAME_CONTRACT_TYPE.astype(str),'PENSION':X.DAYS_EMPLOYED_ANOM.astype(str),'AGE':pd.cut(age,[0,30,40,50,60,100]).astype(str),
 'GENDER':X.CODE_GENDER.astype(str),'INC_TYPE':X.NAME_INCOME_TYPE.astype(str),'TERM_HAT':X.TERM_HAT.fillna(-1).astype(int).astype(str)}
print('| Nhóm | Mức | n | tỉ lệ | vỡ nợ | AUC stack r1 | AUC stack r2 | Chênh |'); print('|---|---|---|---|---|---|---|---|')
for g,v in grp.items():
    v=v.astype(str)
    for lv in sorted(v.unique()):
        m=(v==lv).values
        if m.sum()<1500 or y[m].sum()<60: continue
        a1=A(y[m],s1[m]); a2=A(y[m],s2[m]); print(f'| {g} | {lv} | {m.sum()} | {m.mean():.3f} | {y[m].mean():.3f} | {a1:.4f} | {a2:.4f} | {a2-a1:+.4f} |')

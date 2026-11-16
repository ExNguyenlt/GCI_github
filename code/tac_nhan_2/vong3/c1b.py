# Chan doan OOF vong 2 (khong huan luyen GBDT)
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
from fe_r2 import PRODUCT_KEY
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
pd.set_option('display.width',250)
df=pd.read_pickle(S+'/r2_df.pkl')[0]; X=df.iloc[:n].reset_index(drop=True)
M=['lgb_r2','lgbte_r2','cat_r2','xgb_r2','lrbin_r2','ebm_r2','cat_r1','lgb_r1']
O={m:ld(m,'oof') for m in M}
print('--- AUC'); [print(m,round(A(y,O[m]),5)) for m in M]
print('--- spearman'); print(pd.DataFrame({m:rankdata(O[m]) for m in M}).corr().round(4).to_string())
SEL=['lgbte_r2','cat_r2','xgb_r2','lrbin_r2']
s2=stack(SEL)
# nhom: cat vs lgbte vs stack
age=-X.DAYS_BIRTH/365.25
grp={'EXT_NA':X.EXT_NA.astype(int).astype(str),'CONTRACT':X.NAME_CONTRACT_TYPE.astype(str),'PENSION':X.DAYS_EMPLOYED_ANOM.astype(str),
 'AGE':pd.cut(age,[0,30,40,50,60,100]).astype(str),'GENDER':X.CODE_GENDER.astype(str),'TERM_HAT':X.TERM_HAT.fillna(-1).astype(int).astype(str),
 'PRODCNT':pd.cut(X.PRODUCT_FREQ,[0,30,200,1000,1e9]).astype(str),'EDU':X.NAME_EDUCATION_TYPE.astype(str),'TDR':pd.qcut(X.TERM_DENS_RATIO,4,duplicates='drop').astype(str),
 'GOODS_NA':X.AMT_GOODS_PRICE.isna().astype(str),'INS':(X.AMT_CREDIT>X.AMT_GOODS_PRICE+1).astype(str)}
rows=[]
for g,v in grp.items():
    v=v.astype(object).where(v.notna(),'NA').astype(str)
    for lv in sorted(v.unique()):
        m=(v==lv).values
        if m.sum()<1500 or y[m].sum()<60: continue
        f=lambda o: round(A(y[m],o[m]),4)
        rows.append((g,lv,m.sum(),round(y[m].mean(),3),f(s2),f(O['lgb_r2']),f(O['lgbte_r2']),f(O['cat_r2']),f(O['cat_r1']),f(O['lrbin_r2']),round(f(O['cat_r2'])-f(O['lgbte_r2']),4),round(f(O['cat_r2'])-f(O['cat_r1']),4)))
print(pd.DataFrame(rows,columns=['grp','lvl','n','dr','stack','lgb','lgbte','cat','cat_r1','lrbin','cat-lgbte','cat_r2-r1']).to_string())
# phan ra AUC: between-product vs within-product
pk=PRODUCT_KEY(df)[:n]; z=pd.Series(s2); gm=z.groupby(pk).transform('mean').values
print('AUC stack',round(A(y,s2),5),'AUC cua trung binh stack theo PRODUCT (between)',round(A(y,gm),5),'AUC within (stack - group mean)',round(A(y,s2-gm),5))
# tu tuong quan theo SK_ID_CURR
tr=pd.concat([pd.read_csv(f'/home/user/GCI_github/com/train_part{i}.csv',encoding='utf-8-sig',usecols=['SK_ID_CURR','TARGET']) for i in (1,2)]).reset_index(drop=True)
assert (tr.TARGET.values==y).all()
o=np.argsort(tr.SK_ID_CURR.values); ys=y[o].astype(float); rs=(y-1/(1+np.exp(-s2)))[o]
print('ID sorted? monotone in file:',(np.diff(tr.SK_ID_CURR.values)>0).mean())
for k in (1,5,50,500):
    c=np.convolve(ys,np.ones(2*k+1),'same')-ys; print('k',k,'AUC(y ~ mean y of +-k ID neighbors)',round(A(ys,c),4),' corr resid-neighbor resid',round(np.corrcoef(rs,np.convolve(rs,np.ones(2*k+1),'same')-rs)[0,1],4))
# seed gain
print('--- loi ich seed tu log'); 
import re
for t in ['lgb','lgbte','cat','xgb']:
    L=open(f'{S}/log_r2_{t}.txt').read(); sd=[float(x) for x in re.findall(r'fold \d seed \d ([\d.]+)',L)]; fd=[float(x) for x in re.findall(r'fold \d AUC ([\d.]+)',L)]
    print(t,'mean single-seed fold AUC',round(np.mean(sd),5),'mean seed-avg fold AUC',round(np.mean(fd[-5:]),5))

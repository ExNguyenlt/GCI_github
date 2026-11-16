# Do quan trong nhom dac trung moi (gain + hoan vi theo nhom) tren fold 0, lr 0.05 (nhe)
import sys, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r1 import load_r1
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
df,y,n,cats,ids,G=load_r1()
for c in cats: df[c]=df[c].astype('category')
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; cols=orig+G['IMP']+G['RATE']+G['TERM']
X=df.iloc[:n][cols].reset_index(drop=True)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
P=dict(objective='binary',learning_rate=0.05,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4,seed=0)
a,b=folds[0]; t=time.time()
m=lgb.train(P,lgb.Dataset(X.loc[a],y[a]),5000,valid_sets=[lgb.Dataset(X.loc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
Xb=X.loc[b].reset_index(drop=True); base=A(y[b],m.predict(Xb,num_iteration=m.best_iteration)); print('fold0 AUC',round(base,5),m.best_iteration,f'{time.time()-t:.0f}s')
gn=pd.Series(m.feature_importance('gain'),index=cols); gn=gn/gn.sum()
print('gain share by group:',{k:round(gn[v].sum(),3) for k,v in G.items() if k in('IMP','RATE','TERM')},'orig',round(gn[orig].sum(),3))
print('top 25 gain:'); print(gn.sort_values(ascending=False).head(25).round(4).to_string())
print('new feats gain:'); print(gn[G['IMP']+G['RATE']+G['TERM']].round(4).sort_values(ascending=False).to_string())
rng=np.random.default_rng(0)
def perm(cl,rep=3):
    ds=[]
    for r in range(rep):
        Z=Xb.copy(); idx=rng.permutation(len(Z))
        for c in cl: Z[c]=Z[c].values[idx]
        ds.append(base-A(y[b],m.predict(Z,num_iteration=m.best_iteration)))
    return np.mean(ds)
sets={'IMP':G['IMP'],'RATE':G['RATE'],'TERM':G['TERM'],'RATE+TERM':G['RATE']+G['TERM'],
 'EXT raw+old combos':[c for c in orig if 'EXT' in c or c[:2] in('E1','E2')],
 'ALL EXT (old+IMP)':[c for c in orig if 'EXT' in c or c[:2] in('E1','E2')]+G['IMP'],
 'AMT/CREDIT_ANNUITY old':['AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CREDIT_ANNUITY','CREDIT_GOODS','GOODS_MINUS','TERM_ROUND','EXT_x_CA','EXT_d_CA'],
 'DAYS_BIRTH+age':[c for c in cols if 'BIRTH' in c or c=='AGE_END_LOAN'],
 'ORG/OCC':[c for c in cols if c.startswith('ORGANIZATION') or c.startswith('OCCUPATION') or c=='INC_ORG_OCC_REL'],
 'TERM_HAT,RATE_HAT':['TERM_HAT','RATE_HAT'],'PRODUCT_FREQ,CG_FREQ':['PRODUCT_FREQ','CG_FREQ'],'AGE_END_LOAN':['AGE_END_LOAN'],'OVERPAY_REL':['OVERPAY_REL'],
 'EXT_k_RES':['EXT_SOURCE_1_RES','EXT_SOURCE_2_RES','EXT_SOURCE_3_RES'],'EXT_k_PRED':['EXT_SOURCE_1_PRED','EXT_SOURCE_2_PRED','EXT_SOURCE_3_PRED'],
 'GENDER':['CODE_GENDER','CODE_GENDER_FREQ'],'EDU':['NAME_EDUCATION_TYPE','NAME_EDUCATION_TYPE_FREQ']+[c for c in cols if c.startswith('NAME_EDUCATION_TYPE_')],
}
for k,v in sets.items(): print(f'perm {k:24s} ({len(v):2d} cols) dAUC {perm(v):+.5f}',flush=True)

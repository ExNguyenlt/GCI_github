# OpenFE (Zhang et al., ICML 2023): sinh dac trung tu dong + feature boosting. Chon dac trung CHI bang nhan cua phan train cua fold (a) -> cham tren b.
# python3 ofe.py <fold> [nsub]
from base import *
sys.path.append(S+'/loop/a1r3/pylib')
import types; sys.modules["pandas.core.arrays.arrow.extension_types"]=types.ModuleType("pandas.core.arrays.arrow.extension_types")
from openfe import OpenFE, transform
fo=int(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 0
num=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','CNT_FAM_MEMBERS','CNT_CHILDREN','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_MON','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1_FILL','EXT_SOURCE_3_FILL','EXTF_MEAN','RATE_HAT','TERM_HAT','CREDIT_ANNUITY','CREDIT_GOODS','PT_e','PT_e3','PT_age','PT_emp']
cat=list(cats)+['REGION_RATING_CLIENT']
D=df[num+cat].copy()
for c in cat: D[c]=pd.factorize(D[c].astype(str))[0]
D=D.astype(float)
a,b=folds[fo]; p=np.load(f'inner_{fo}.npy'); init=pd.DataFrame(np.log(p/(1-p)))
Xa=D.iloc[a].reset_index(drop=True); ya=pd.DataFrame({'y':y[a]})
if NS:
    rs=np.random.RandomState(0).choice(len(a),NS,replace=False); Xa=Xa.iloc[rs].reset_index(drop=True); ya=ya.iloc[rs].reset_index(drop=True); init=init.iloc[rs].reset_index(drop=True)
t=time.time(); ofe=OpenFE()
feats=ofe.fit(data=Xa,label=ya,init_scores=init,task='classification',metric='auc',categorical_features=cat,n_data_blocks=8,min_candidate_features=1000,stage2_params={'n_estimators':300,'importance_type':'gain','num_leaves':16,'seed':1,'n_jobs':int(os.environ.get('NJ','2')),'learning_rate':0.05,'colsample_bytree':0.3,'verbose':-1},n_jobs=int(os.environ.get('NJ','2')),seed=1,verbose=False)
print('openfe fit',time.time()-t,'n feats',len(feats),flush=True)
from openfe import tree_to_formula
for f in feats[:40]: print(tree_to_formula(f))
K=40; tr,te=transform(D.iloc[:n].reset_index(drop=True),D.iloc[n:].reset_index(drop=True),feats[:K],n_jobs=2)
new=[c for c in tr.columns if c not in D.columns]; out=pd.concat([tr[new],te[new]],ignore_index=True); out.columns=[f'OFE{i}' for i in range(len(new))]
out.to_pickle(f'ofe_f{fo}.pkl'); print(out.shape)

# Vong 3. Huan luyen tang 1 voi SO VONG CO DINH (khong early stopping), ghi duong cong AUC valid + luu mo hinh, de bao duoc CA HAI con so (L3.4/R22):
#   (ES)  du doan tai so vong "early stopping" mo phong (patience 300 tren fold cham, nhu vong 0-2)
#   (FIX) du doan tai so vong = trung binh so vong ES cua 4 fold KHAC (moi seed) -> khong dung nhan fold cham de chon so vong
# python3 -I run_r3.py train   <lgb|cat> <spec> <seed> <tag> [fold,fold,...]     -> r3m/<tag>_s<seed>_f<f>.{npz,model}   (bo qua neu da co)
# python3 -I run_r3.py collect <lgb|cat> <spec> <seed,seed,...> <tag>            -> oof_<tag>.npy, pt_<tag>.npy (ES) ; oof_<tag>_fix.npy, pt_<tag>_fix.npy (FIX)
# spec: +GRP-GRP~FLAG...   FLAG: TE (TE PRODUCT long fold R14) | PCAT (PRODUCT categorical, chi CatBoost) | PS50 | ET | LT | DART | LR<x> | NR<x> | NI<thr> (null importance trong fold)
import sys, os, re, time, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from fe_r3 import load_r3, PRODUCT_KEY, nested_te
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as A
mode,which,spec,seeds,tag=sys.argv[1],sys.argv[2],sys.argv[3],[int(s) for s in sys.argv[4].split(',')],sys.argv[5]
FL=[int(f) for f in sys.argv[6].split(',')] if len(sys.argv)>6 else [0,1,2,3,4]
df,y,n,cats,ids,G=load_r3()
newc=set(sum(G.values(),[])); orig=[c for c in df.columns if c not in newc]; cols=orig+G['IMP']+G['RATE']+G['TERM']
parts=spec.split('~'); flags=parts[1:]
for sgn,nm in re.findall(r'([+-])([A-Za-z0-9_]+)',parts[0]):
    cc=G[nm] if nm in G else [nm]
    if sgn=='+': cols+=[c for c in cc if c not in cols]
    else: cols=[c for c in cols if c not in cc]
cats=list(cats); PROD=PRODUCT_KEY(df)
if 'PCAT' in flags:
    vc=pd.Series(PROD).value_counts(); df['PRODUCT_CAT']=np.where(pd.Series(PROD).map(vc).values<30,'rare',PROD); cols.append('PRODUCT_CAT'); cats.append('PRODUCT_CAT')
fv=lambda pre,d: next((float(f[len(pre):]) for f in flags if re.fullmatch(pre+r'[0-9.]+',f)),d)
DART='DART' in flags; PAT=300
LR=fv('LR',0.01 if which=='lgb' else 0.03); NR=int(fv('NR',3300 if which=='lgb' else 2100)); NI=next((float(f[2:]) for f in flags if re.fullmatch(r'NI-?[0-9.]+',f)),None)
X=df.iloc[:n][cols].reset_index(drop=True); T=df.iloc[n:][cols].reset_index(drop=True)
if which=='lgb':
    for c in cats: X[c]=X[c].astype('category'); T[c]=pd.Categorical(T[c],categories=X[c].cat.categories)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
def build(f):
    a,b=folds[f]; Xa,Xb,Tt=X.iloc[a].reset_index(drop=True),X.iloc[b].reset_index(drop=True),T.copy()
    if 'TE' in flags:
        ta,(tb,tt)=nested_te(PROD,y,a,[b,np.arange(n,len(df))],10); Xa['TE_PRODUCT']=ta; Xb['TE_PRODUCT']=tb; Tt['TE_PRODUCT']=tt
    if 'LT' in flags:
        num=[c for c in Xa.columns if c not in cats]; med=Xa[num].median(); Xa[num]=Xa[num].fillna(med); Xb[num]=Xb[num].fillna(med); Tt[num]=Tt[num].fillna(med)
    if NI is not None:      # Null Importance: chi dung nhan phan train cua fold
        import lightgbm as lgb
        Pn=dict(objective='binary',learning_rate=0.1,num_leaves=14,max_depth=4,min_child_samples=50,feature_fraction=0.2,bagging_fraction=0.7,bagging_freq=1,lambda_l1=10,cat_smooth=40,max_cat_to_onehot=4,verbose=-1,n_jobs=4)
        act=sum(lgb.train(dict(Pn,seed=s),lgb.Dataset(Xa,y[a]),150).feature_importance('gain') for s in range(3))/3; rs=np.random.RandomState(0)
        nul=np.array([lgb.train(dict(Pn,seed=r),lgb.Dataset(Xa,rs.permutation(y[a])),150).feature_importance('gain') for r in range(20)])
        keep=[c for c,s in zip(Xa.columns,np.log((1e-10+act)/(1+np.percentile(nul,75,axis=0)))) if s>NI]; print('NI fold',f,'giu',len(keep),'/',Xa.shape[1],flush=True)
        Xa,Xb,Tt=Xa[keep],Xb[keep],Tt[keep]
    return a,b,Xa,Xb,Tt
def es_iter(curve,pat=PAT):
    best=-1; bi=0
    for i,v in enumerate(curve):
        if v>best: best=v; bi=i
        elif i-bi>=pat: break
    return bi+1
def predict(m,Xd,k):
    if which=='lgb': return m.predict(Xd,num_iteration=k)
    return m.predict_proba(Xd,ntree_end=k)[:,1]
def fn(s,f,ext): return f'{S}/r3m/{tag}_s{s}_f{f}.{ext}'
if mode=='train':
    t0=time.time()
    for f in FL:
        todo=[s for s in seeds if not os.path.exists(fn(s,f,'npz'))]
        if not todo: continue
        a,b,Xa,Xb,Tt=build(f)
        for s in todo:
            if which=='lgb':
                import lightgbm as lgb
                P=dict(objective='binary',metric='auc',learning_rate=LR,num_leaves=16,max_depth=6,min_child_samples=100,feature_fraction=0.3,bagging_fraction=0.8,bagging_freq=1,lambda_l1=2,lambda_l2=10,min_gain_to_split=0.01,cat_smooth=50,cat_l2=20,max_cat_to_onehot=4,verbose=-1,n_jobs=4,seed=s)
                if 'PS50' in flags: P['path_smooth']=50
                if 'ET' in flags: P['extra_trees']=True
                if 'LT' in flags: P.update(linear_tree=True,linear_lambda=10.0)
                if DART: P.update(boosting='dart',drop_rate=0.1,skip_drop=0.5)
                ev={}; m=lgb.train(P,lgb.Dataset(Xa,y[a]),NR,valid_sets=[lgb.Dataset(Xb,y[b])],callbacks=[lgb.record_evaluation(ev)]); curve=np.array(ev['valid_0']['auc']); m.save_model(fn(s,f,'model'))
            else:
                from catboost import CatBoostClassifier
                m=CatBoostClassifier(iterations=NR,learning_rate=LR,depth=6,l2_leaf_reg=5,eval_metric='AUC',thread_count=4,verbose=0,random_seed=s)
                m.fit(Xa,y[a],cat_features=cats,eval_set=(Xb,y[b]),use_best_model=False); curve=np.array(m.evals_result_['validation']['AUC']); m.save_model(fn(s,f,'model'))
            k=NR if DART else es_iter(curve)      # DART: cay sau doi trong so cay truoc -> khong cat duoc; dung du NR vong (co dinh, khong chon tren fold cham)
            vb=predict(m,Xb,k); vt=predict(m,Tt,k)
            np.savez(fn(s,f,'npz'),vb=vb,vt=vt,k=k,curve=curve)
            print(tag,'fold',f,'seed',s,'ES AUC',round(A(y[b],vb),5),'k',k,'| AUC@NR',round(float(curve[-1]),5),f'{time.time()-t0:.0f}s',flush=True)
else:
    K={(s,f):int(np.load(fn(s,f,'npz'))['k']) for s in seeds for f in range(5)}
    oof=np.zeros(n); pt=np.zeros(len(T)); oofx=np.zeros(n); ptx=np.zeros(len(T)); kfix={}
    for f in range(5):
        a,b,Xa,Xb,Tt=build(f); kf=int(round(np.mean([K[s,g] for s in seeds for g in range(5) if g!=f]))); kfix[f]=kf
        for s in seeds:
            z=np.load(fn(s,f,'npz')); oof[b]+=z['vb']/len(seeds); pt+=z['vt']/len(seeds)/5
            if DART: oofx[b]+=z['vb']/len(seeds); ptx+=z['vt']/len(seeds)/5; continue
            if which=='lgb':
                import lightgbm as lgb; m=lgb.Booster(model_file=fn(s,f,'model'))
            else:
                from catboost import CatBoostClassifier; m=CatBoostClassifier(); m.load_model(fn(s,f,'model'))
            oofx[b]+=predict(m,Xb,kf)/len(seeds); ptx+=predict(m,Tt,kf)/len(seeds)/5
    fl=lambda o:[round(A(y[b],o[b]),5) for a,b in folds]
    print(tag,'ES  OOF AUC',round(A(y,oof),5),'folds',fl(oof),'k',{f:[K[s,f] for s in seeds] for f in range(5)})
    print(tag,'FIX OOF AUC',round(A(y,oofx),5),'folds',fl(oofx),'kfix',kfix,flush=True)
    np.save(f'{S}/oof_{tag}.npy',oof); np.save(f'{S}/pt_{tag}.npy',pt); np.save(f'{S}/oof_{tag}_fix.npy',oofx); np.save(f'{S}/pt_{tag}_fix.npy',ptx)

# Vong 3 (cuoi): so sanh ensemble (rank-avg / stacking LR long fold) va ghi submission.   python3 -I ens_final_r3.py [write <set_name>]
# Moi OOF tang 1 cung folds rs=42; meta-LR fit tren 4 fold, cham fold con lai (ens_r2.stack). Ban "_fix" = tang 1 so vong co dinh (L3.4/R22).
import sys, os, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
from scipy.stats import spearmanr
have=lambda m: os.path.exists(f'{S}/oof_{m}.npy')
R2=['lgbte_r2','cat_r2','xgb_r2','lrbin_r2']; NEW=[m for m in ['lgbte_r3','cat_r3','lrbin_r3','dart_r3','et_r3','lt_r3','tabm_r3','tabm3_r3','catn_r3','lgbte_r3_fix','cat_r3_fix'] if have(m)]
if all(have(m) for m in ['tabm_r3','tabmB_r3','tabmC_r3']) and not have('tabm3_r3'):
    for k in ('oof','pt'): np.save(f'{S}/{k}_tabm3_r3.npy',np.mean([ld(m,k) for m in ['tabm_r3','tabmB_r3','tabmC_r3']],0))
    NEW.insert(7,'tabm3_r3')
print('--- mo hinh don'); 
for m in R2+NEW: rep(m,ld(m,'oof'))
ref=rep('MOC: stack r2 (lgbte+cat+xgb+lrbin)',stack(R2))
SETS={'r2_noXGB (L3.1)':('stack',['lgbte_r2','cat_r2','lrbin_r2'])}
core=[m for m in ['lgbte_r3','cat_r3','lrbin_r3'] if have(m)]
if len(core)==3:
    SETS['rank_core3']=('rank',core); SETS['stack_core3']=('stack',core)
    SETS['stack_core3+xgb_r2']=('stack',core+['xgb_r2'])
    SETS['stack_core3+r2(lgbte,cat)']=('stack',core+['lgbte_r2','cat_r2'])
    SETS['stack_lgbte_r3+cat_r2+lrbin_r2']=('stack',['lgbte_r3','cat_r2','lrbin_r2'])
    SETS['stack_lgbte_r2+cat_r3+lrbin_r2']=('stack',['lgbte_r2','cat_r3','lrbin_r2'])
    div=[m for m in ['dart_r3','et_r3','lt_r3','tabm3_r3'] if have(m)]
    if have('tabm_r3'): SETS['stack_core3+tabm_r3(1 seed)']=('stack',core+['tabm_r3'])
    for d in div: SETS[f'stack_core3+{d}']=('stack',core+[d])
    if div: SETS['stack_core3+alldiv']=('stack',core+div); SETS['rank_core3+alldiv']=('rank',core+div)
    if have('lgbte_r3_fix') and have('cat_r3_fix'):
        # cat_r3_fix chi co 3 seed moi (2,3,4) -> cap so sanh cong bang cho do lac quan ES la ban ES 3 seed (catn_r3)
        SETS['stack_ES_cat3seed']=('stack',['lgbte_r3','catn_r3','lrbin_r3']); SETS['stack_FIX_cat3seed']=('stack',['lgbte_r3_fix','cat_r3_fix','lrbin_r3'])
        if have('tabm3_r3'): SETS['stack_ES_cat3seed+tabm3']=('stack',['lgbte_r3','catn_r3','lrbin_r3','tabm3_r3']); SETS['stack_FIX_cat3seed+tabm3']=('stack',['lgbte_r3_fix','cat_r3_fix','lrbin_r3','tabm3_r3'])
res={}
for k,(kind,names) in SETS.items():
    if kind=='rank': o,p=rankavg(names); c=None
    else: o,p,c=stack(names,(),test=True)
    res[k]=(o,p); rep(k,o,ref)
    if c is not None: print('      coef',dict(zip(names,[float(x) for x in c])))
if 'stack_core3' in res:
    b=fa(res['stack_core3'][0]); print('--- chenh so voi stack_core3')
    for k in res:
        d=fa(res[k][0])-b; print(f'  {k:36s}: {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5')
    rg=res['stack_core3'][0]; print('spearman vs stack_core3:',{m:round(spearmanr(ld(m,'oof'),rg)[0],4) for m in NEW})
if len(sys.argv)>2 and sys.argv[1]=='write':
    k=sys.argv[2]; o,p=res[k]; ids=np.load(S+'/ids.npy')
    ss=pd.read_csv('/home/user/GCI_github/com/sample_submission.csv',encoding='utf-8-sig')
    assert len(ss)==len(p)==len(ids)==61500 and (ss.SK_ID_CURR.values==ids).all()
    pr=rankdata(p)/len(p); sub=pd.DataFrame({'SK_ID_CURR':ss.SK_ID_CURR.values,'TARGET':pr}); assert sub.TARGET.notna().all()
    sub.to_csv(S+'/loop/sub_round3.csv',index=False,encoding='utf-8'); np.save(S+'/oof_final_r3.npy',o); np.save(S+'/pt_final_r3.npy',p)
    print('wrote',S+'/loop/sub_round3.csv',k,sub.shape,'OOF',round(A(y,o),5),'| spearman test vs sub_round2',round(spearmanr(pr,pd.read_csv(S+'/loop/sub_round2.csv').TARGET.values)[0],4))

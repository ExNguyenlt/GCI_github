# Vong 2: so sanh ensemble (rank-avg / stacking LR long nhau, co/khong cot TE PRODUCT o meta) va ghi submission.
# python3 -I ens_final_r2.py [write <set_name>]
import sys, os, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
from fe_r2 import PRODUCT_KEY
from scipy.stats import spearmanr
df=pd.read_pickle(S+'/r2_df.pkl')[0]; TE=te_extra(PRODUCT_KEY(df),10)
have=lambda m: os.path.exists(f'{S}/oof_{m}.npy')
# cot "mo hinh rieng theo nhom" (L2.4b): lgb_r2 voi dong thuoc nhom duoc thay bang du doan cua mo hinh rieng
for g in ['old','rev','young','ext2na']:
    if have(f'sub_{g}_r2') and have('lgb_r2'):
        o=ld('lgb_r2','oof').copy(); s=ld(f'sub_{g}_r2','oof'); m=~np.isnan(s); o[m]=s[m]; np.save(f'{S}/oof_lgbsub_{g}_r2.npy',o)
        p=ld('lgb_r2','pt').copy(); s=ld(f'sub_{g}_r2','pt'); m=~np.isnan(s); p[m]=s[m]; np.save(f'{S}/pt_lgbsub_{g}_r2.npy',p)
G1=['lgb_r1','cat_r1','xgb_r1']; G2=[m for m in ['lgb_r2','cat_r2','xgb_r2'] if have(m)]
allm=G1+['lrbin_r1']+G2+[m for m in ['lrbin_r2','ebm_r2','lgbte_r2','catnp_r2'] if have(m)]
for m in allm: rep(m,ld(m,'oof'))
rg=rankavg(G2)[0]; print('spearman vs rank-avg G2:',{m:round(spearmanr(ld(m,'oof'),rg)[0],4) for m in allm if m not in G1})
ref=rep('MOC: stack r1 (G1+lrbin_r1)',stack(G1+['lrbin_r1']))
SETS={'rank_G2':('rank',G2,[]),'stack_G2':('stack',G2,[]),'stack_G2+TE':('stack',G2,[TE]),
      'stack_G2+lrbin_r1':('stack',G2+['lrbin_r1'],[]),'stack_G2+lrbin_r2':('stack',G2+['lrbin_r2'],[]),
      'stack_G2+lrbin_r2+TE':('stack',G2+['lrbin_r2'],[TE]),'stack_G2+lrbin_r1+lrbin_r2+TE':('stack',G2+['lrbin_r1','lrbin_r2'],[TE]),
      'stack_G2+G1+lrbin_r2+TE':('stack',G2+G1+['lrbin_r2'],[TE])}
for x in ['ebm_r2','lgbte_r2','catnp_r2']:
    if have(x): SETS[f'stack_G2+lrbin_r2+{x}']=('stack',G2+['lrbin_r2',x],[]); SETS[f'stack_G2+lrbin_r2+{x}+TE']=('stack',G2+['lrbin_r2',x],[TE])
if have('lgbte_r2'): SETS['stack_lgbte+cat+xgb+lrbin_r2']=('stack',['lgbte_r2']+G2[1:]+['lrbin_r2'],[]); SETS['stack_lgbte+cat+xgb+lrbin_r2+TE']=('stack',['lgbte_r2']+G2[1:]+['lrbin_r2'],[TE])
for g in ['old','rev','young','ext2na']:
    if have(f'lgbsub_{g}_r2'): SETS[f'stack_G2+lrbin_r2+sub_{g}']=('stack',G2+['lrbin_r2',f'lgbsub_{g}_r2'],[])
res={}
for k,(kind,names,ex) in SETS.items():
    if kind=='rank': o,p=rankavg(names); c=None
    else: o,p,c=stack(names,ex,test=True)
    res[k]=(o,p); rep(k,o,ref); 
    if c is not None: print('      coef',dict(zip(names+['TE']*len(ex),c)))
base='stack_G2+lrbin_r2'
if base in res:
    b=fa(res[base][0])
    for k in res:
        d=fa(res[k][0])-b; print(f'  {k:40s} - {base}: {np.round(d,5)} mean {d.mean():+.5f} pos {(d>0).sum()}/5')
if len(sys.argv)>2 and sys.argv[1]=='write':
    k=sys.argv[2]; o,p=res[k]; ids=np.load(S+'/ids.npy')
    ss=pd.read_csv('/home/user/GCI_github/com/sample_submission.csv'); ss.columns=[c.replace('﻿','') for c in ss.columns]
    assert len(ss)==len(p)==len(ids) and (ss.SK_ID_CURR.values==ids).all()
    pr=rankdata(p)/len(p); sub=pd.DataFrame({'SK_ID_CURR':ss.SK_ID_CURR.values,'TARGET':pr}); assert sub.TARGET.notna().all()
    sub.to_csv(S+'/loop/sub_round2.csv',index=False); np.save(S+'/oof_final_r2.npy',o); np.save(S+'/pt_final_r2.npy',p)
    print('wrote',S+'/loop/sub_round2.csv',k,sub.shape,'OOF',round(A(y,o),5))

import sys, numpy as np, pandas as pd
S='/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad'; sys.path.insert(0,S)
from ens_r2 import *
from fe_r2 import PRODUCT_KEY
df=pd.read_pickle(S+'/r2_df.pkl')[0]; TE=te_extra(PRODUCT_KEY(df),10)
ref=rep('lgb_r1 don',lg(ld('lgb_r1','oof')))
rep('lgb_r2 don',lg(ld('lgb_r2','oof')),ref)
r=rep('stack lgb_r2',stack(['lgb_r2']))
rep('stack lgb_r2 + TE meta',stack(['lgb_r2'],[TE]),r)
r=rep('stack lgb_r2+lrbin_r2',stack(['lgb_r2','lrbin_r2']))
rep('stack lgb_r2+lrbin_r2 + TE meta',stack(['lgb_r2','lrbin_r2'],[TE]),r)

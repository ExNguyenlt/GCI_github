from base import *
ext=[c for c in BASE if c in ('EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1_FILL','EXT_SOURCE_2_FILL','EXT_SOURCE_3_FILL','EXTF_MEAN','EXTF_MIN','EXTF_MAX','EXTF_PROD','EXTF_W')]
mc=[-1 if c in ext else 0 for c in BASE]
V={'mono_ext':dict(monotone_constraints=mc,monotone_constraints_method='advanced'),'unbalance':dict(scale_pos_weight=3.0),'pathsmooth':dict(path_smooth=50),
 'goss':dict(data_sample_strategy='goss',bagging_fraction=1.0,bagging_freq=0),'maxbin1023':dict(max_bin=1023)}
for v in sys.argv[1:]: screen(v,P2=V[v])

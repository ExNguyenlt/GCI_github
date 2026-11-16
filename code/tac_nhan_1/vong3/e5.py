from base import *
F=pd.read_pickle('f2.pkl'); g=[c for c in F.columns if c.startswith('PR_')]
FS=(2,3,4)   # fold 0/1 da co trong scr.json (BASE, PRres_all, pathsmooth, distill_0.6)
screen('BASE_f234',FS=FS,ref='BASE_f234')
screen('PRres_all_f234',F[g],FS=FS,ref='BASE_f234')
screen('pathsmooth_f234',P2=dict(path_smooth=50),FS=FS,ref='BASE_f234')

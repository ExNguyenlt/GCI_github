from base import *
F=pd.read_pickle('f2.pkl')
g=lambda pre:[c for c in F.columns if c.startswith(pre)]
screen('PRres_rate',F[['PR_RATEHAT_PRED','PR_RATEHAT_RES','PR_LCA_PRED','PR_LCA_RES']])
screen('PRres_all',F[g('PR_')])
screen('PRres_inc',F[['PR_LINC_PRED','PR_LINC_RES','PR_EMP_PRED','PR_EMP_RES']])
screen('mod7',F[g('M7_')])

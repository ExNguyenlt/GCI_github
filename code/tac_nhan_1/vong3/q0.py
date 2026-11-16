import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score as A
D='/home/user/GCI_github/com/'
a=pd.read_csv(D+'train_part1.csv',encoding='utf-8-sig'); b=pd.read_csv(D+'train_part2.csv',encoding='utf-8-sig'); t=pd.read_csv(D+'test.csv',encoding='utf-8-sig')
print(list(a.columns)); print(len(a),len(b),len(t)); print(a.TARGET.mean(),b.TARGET.mean())
tr=pd.concat([a,b],ignore_index=True)
print('id head',tr.SK_ID_CURR.head(8).tolist(), t.SK_ID_CURR.head(8).tolist(), 'sorted?',tr.SK_ID_CURR.is_monotonic_increasing, t.SK_ID_CURR.is_monotonic_increasing)
print('id range',tr.SK_ID_CURR.min(),tr.SK_ID_CURR.max(),t.SK_ID_CURR.min(),t.SK_ID_CURR.max())
y=tr.TARGET.values
# file-order autocorr
for lag in (1,2,5): print('lag',lag,np.corrcoef(y[:-lag],y[lag:])[0,1])
# rolling mean excluding self in file order
s=pd.Series(y); 
for w in (50,500,5000):
    r=(s.rolling(w,center=True,min_periods=1).sum()-s)/(s.rolling(w,center=True,min_periods=1).count()-1); print('roll file',w,A(y,r))
o=np.argsort(tr.SK_ID_CURR.values); so=pd.Series(y[o])
for w in (50,500,5000):
    r=(so.rolling(w,center=True,min_periods=1).sum()-so)/(so.rolling(w,center=True,min_periods=1).count()-1); print('roll id',w,A(y[o],r))
# feature drift along id: corr of features with id rank
num=tr.select_dtypes('number').drop(columns=['TARGET'])
print(num.corrwith(tr.SK_ID_CURR,method='spearman').abs().sort_values(ascending=False).head(6))
print(num.corrwith(pd.Series(np.arange(len(tr))),method='spearman').abs().sort_values(ascending=False).head(6))

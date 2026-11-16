from base import *
O=pd.read_pickle('ofe0/ofe_f0.pkl')
screen('BASEf0',FS=(0,),SS=(0,1,2,3),ref='BASEf0')
for k in (10,40): screen(f'ofe{k}_f0',O.iloc[:,:k],FS=(0,),SS=(0,1,2,3),ref='BASEf0')

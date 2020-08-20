import pandas as pd
import numpy as np
import os
from scipy.io import arff
from pandas import DataFrame
import math
import threading
import time
data = arff.loadarff('CandidateData.arff')
df = pd.DataFrame(data[0])
df1=df.copy()
for i in range(0,24):
    df1['optimal'][i]=math.ceil(df['optimal'][i]/(200*0.94))
    df1['ar'][i]=math.ceil(df['ar'][i]/(200*0.94))
    df1['arima'][i]=math.ceil(df['arima'][i]/(200*0.94))
    df1['ma'][i]=math.ceil(df['ma'][i]/(200*0.94))
    df1['lr'][i]=math.ceil(df['lr'][i]/(200*0.94))
    df1['svm'][i]=math.ceil(df['svm'][i]/(200*0.94))
    df1['krr'][i]=math.ceil(df['krr'][i]/(200*0.94))
print(df1)
df1.to_csv('no_of_instances.csv')
#assume we have initially 20 server 
# according to paper we have assumed the following values

cdt=10
sdr=0.40
R_min=5
R_current=20

for i in range(0,24):
    R_estimated=int(df1['ar'][i]);
    if R_current==R_estimated:
        os.system('sudo docker service ls')
    elif R_estimated > R_current:
        cmd='sudo docker service scale prod_awesome='+str(R_estimated)
        os.system(cmd)
        os.system('sudo docker service ls')
        R_current = R_estimated 
        cdt=10
    elif cdt==0:
        cdt=10
        R_estimated=math.floor((R_current-R_estimated)*(1-sdr))
        R_estimated=math.max(R_current-R_estimated,R_min)
        cmd='sudo docker service scale prod_awesome='+str(R_estimated)
        R_current = R_estimated 
        os.system(cmd)
        os.system('sudo docker service ls')
    else: 
        cdt=cdt-1
    time.sleep(60)

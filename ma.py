import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#matplotlib inline
from scipy.io import arff
from matplotlib.pylab import rcParams
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,NuSVR
import scipy
from sklearn.preprocessing import StandardScaler
from math import sqrt
from scipy.io import arff
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
rcParams['figure.figsize'] = 15, 6
data = arff.loadarff('input-data.arff')
train= pd.DataFrame(data[0])

train=pd.DataFrame(train['hits'])

out_data= arff.loadarff('test-data.arff')
#print(data)
test= pd.DataFrame(out_data[0])
test=pd.DataFrame(test['hits'])


data=train.append(test,ignore_index = True)
data=pd.DataFrame(data['hits'])


scaler = StandardScaler()
x_t=data.values #converting to numpy
data=pd.DataFrame(scaler.fit_transform(data))
 


scalery = StandardScaler()
y_tt=test.values
y_tt=pd.DataFrame(scalery.fit_transform(y_tt))

train_len=672
test_len=24


df1 = data.copy()

df1['moving_avg_forecast'] = data.rolling(3).mean()

train=df1[0:train_len] 
tst=df1[train_len:]


y_t=tst.copy()
print(y_t['moving_avg_forecast'])

y_t=pd.DataFrame(y_t['moving_avg_forecast'])
y_ttt=y_t.values
y_t=pd.DataFrame(scalery.inverse_transform(y_ttt)) #inversing 


r2 = r2_score(y_tt,y_ttt )
print(r2)


time=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
time=pd.DataFrame(time)

plt.plot(time,test,color='red',marker='+',label='Real Value')

plt.plot(time,y_t,color='blue',marker='.',label='Forecast #2(Moving Average)')
plt.xlabel('Time') 
# naming the y axis 
plt.ylabel('Request/s(hourly average)') 


plt.legend()
plt.show()
# to handle the NaN case in order to calculate the RMSE
#print(df1);
rmse = np.sqrt(mean_squared_error(y_tt, y_ttt))
rmse = round(rmse, 3)
print("Root mean square error is : ",rmse)


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_tt, y_ttt)
print("Mean absolute error is :" ,mae)
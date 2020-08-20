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

#input data
data = arff.loadarff('input-data.arff')
series = pd.DataFrame(data[0])

temps=pd.DataFrame(series['hits'])

#Test data
data_test = arff.loadarff('test-data.arff')
series_test = pd.DataFrame(data_test[0])
y_test=pd.DataFrame(series_test['hits'])

temps=temps.append(y_test,ignore_index = True)


dataframe=concat([temps.shift(24),temps.shift(23),temps.shift(22),temps.shift(21),temps.shift(20),temps.shift(19),temps.shift(18),temps.shift(17),temps.shift(16),temps.shift(15),temps.shift(14),temps.shift(13),temps.shift(12),temps.shift(11),temps.shift(10),temps.shift(9),temps.shift(8),temps.shift(7),temps.shift(6),temps.shift(5),temps.shift(4),temps.shift(3),temps.shift(2),temps.shift(1)], axis=1)
dataframe.columns = ['me', 'mx', 't+1','h','g','hth','w','e','r','t','y','i','o','p','a','s','d','m','g','ss','ww','qww','rt','out']


df=DataFrame()
df=dataframe[24:672]
x_test=DataFrame()
x_test=dataframe[672:]
x_train=DataFrame()
x_train=df
y_train=DataFrame()
y_train=pd.DataFrame(temps[24:672])

x_t=x_train.values
y_t=y_train.values



scaler = StandardScaler()
x_t=x_train.values #converting to numpy
x_train=pd.DataFrame(scaler.fit_transform(x_t))
x_te=x_test.values#converting to numpy
x_test=pd.DataFrame(scaler.fit_transform(x_te))

scalery = StandardScaler()
y_t=y_train.values#converting to numpy
y_train=pd.DataFrame(scalery.fit_transform(y_t))
y_tt=y_test.values
y_tt=pd.DataFrame(scalery.fit_transform(y_tt))

y= pd.DataFrame(KernelRidge(alpha=1,coef0=1, gamma=0.02, kernel='rbf',
        kernel_params={"C":1.9, "epsilon":0.027}).fit(x_train, y_train).predict(x_test))



y.columns=['hits']

y_te=y.values
y=pd.DataFrame(scalery.inverse_transform(y_te)) #inversing 



timestamp=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
time=pd.DataFrame(timestamp)
time.columns=['time']







plt.plot(time['time'],y_test,color='red',marker='+')

plt.plot(time['time'],y,color='blue',marker='.')
plt.show()

r2 = r2_score(y_test, y)
print(r2)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_tt,y_te)
mse = mean_squared_error(y_tt.values, y_te)
rmse = np.sqrt(mse)
print(mae)
print(rmse)

mae=mean_absolute_error(y_test,y)
mse = mean_squared_error(y_test, y)
rmse = np.sqrt(mse)
print(mae)
print(rmse)
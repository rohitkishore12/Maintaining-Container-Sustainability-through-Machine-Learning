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


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0, 1)) 
scaler = scaler.fit(x_t)
x_train=scaler.transform(x_t)
x_train=pd.DataFrame(x_train)

x_te=x_test.values
scaler1 = MinMaxScaler(feature_range =(0, 1)) 
scaler1 = scaler1.fit(x_te)
x_test=scaler1.transform(x_te)
x_test=pd.DataFrame(x_test)


scalery = MinMaxScaler(feature_range =(0, 1)) 
scalery = scalery.fit(y_t)
y_train=scalery.transform(y_t)
y_train=pd.DataFrame(y_train)
print(y_train)

model = SVR(kernel='rbf',gamma=0.002,C=10,epsilon=.027)
model.fit(x_train, y_train)
yt=pd.DataFrame(model.predict(x_test))





scalery1=MinMaxScaler(feature_range =(0, 1)) 
scalery1 = scalery1.fit(y_test)
y=scalery1.inverse_transform(yt)
y=pd.DataFrame(y)
y.columns=['hits']


timestamp=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
time=pd.DataFrame(timestamp)
time.columns=['time']




# model evaluation


fig, ax1 = plt.subplots()
color = 'tab:red'
#svr
y1_container = [45, 40, 38, 37, 41, 43, 43, 50, 53, 61, 66, 70, 69, 70, 70, 69, 67, 63, 56, 53, 50, 49, 46, 42]
color1 = 'tab:red'

#krr
y2_container = [46, 40, 38, 37, 40, 43, 43, 50, 52, 60, 65, 69, 68, 69, 69, 68, 66, 63, 57, 54, 51, 49, 46, 42]
color2 = 'tab:brown'
#lr
y3_container = [45, 40, 37, 36, 42, 44, 43, 50, 52, 63, 67, 72, 69, 71, 71, 70, 67, 63, 54, 53, 50, 50, 47, 42]
color3 = 'tab:green'
#arima
y4_container = [59, 49, 45, 44, 44, 45, 47, 48, 49, 51, 52, 52, 53, 53, 55, 57, 54, 55, 57, 58, 59, 58, 55, 54]
color4 = 'tab:purple'
#ar
y5_container = [37, 36, 36, 38, 41, 43, 46, 49, 54, 57, 60, 60, 60, 61, 62, 62, 62, 61, 58, 55, 52, 50, 48, 45]
color5 = 'tab:pink'
#ma
y6_container = [44, 37, 34, 35, 44, 50, 51, 48, 54, 58, 70, 73, 74, 70, 69, 68, 66, 60, 56, 53, 53, 56, 55, 51]
color6 = 'tab:orange'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('No. of replicas', color=color)
ax1.plot(time['time'], y1_container, color=color1,label='SVR')
ax1.plot(time['time'], y2_container, color=color2,label='KRR')
ax1.plot(time['time'], y3_container, color=color3,label='LR')
ax1.plot(time['time'], y5_container, color=color5,label='AR')
ax1.plot(time['time'], y4_container, color=color4,label='ARIMA')
ax1.plot(time['time'], y6_container, color=color6,label='MA')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Requests per seconds ', color=color)  # we already handled the x-label with ax1
ax2.plot(time['time'], y, color=color,label='RPS')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper left')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


r2 = r2_score(y_test, y)
print(r2)

from sklearn.metrics import mean_absolute_error


mae=mean_absolute_error(y_test,y)
mse = mean_squared_error(y_test, y)
rmse = np.sqrt(mse)
print(mae)
print(rmse)


import pandas as pd 
import numpy as np 
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
from pandas.plotting import lag_plot
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.io import arff
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
import datetime
import calendar
import seaborn as sns
from statsmodels.tsa.ar_model import AR


#rcParams['figure.figsize'] = 15, 6
data = arff.loadarff('input-data.arff')
#print(data)
data=pd.DataFrame(data[0])
train=data['hits'].values


out_data= arff.loadarff('test-data.arff')
out_data=pd.DataFrame(out_data[0])
test=out_data['hits'].values 

model = AR(train)
model_fit = model.fit()
model_fit.k_ar=24
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

time=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

time=pd.DataFrame(time)



plt.plot(time,test,color='red',marker='+',label='Real Value')

plt.plot(time,predictions,color='blue',marker='.',label='Forecast #1(AR)')
plt.xlabel('Time') 
# naming the y axis 
plt.ylabel('Request/s(hourly average)') 


plt.legend()
plt.show()



scalery = StandardScaler()
test=pd.DataFrame(test)#converting to numpy
test=scalery.fit_transform(test.values)
predictions=pd.DataFrame(predictions)
predictions=scalery.fit_transform(predictions.values)


r2 = r2_score(test, predictions)
print(r2)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(test,predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(mae)
print(rmse)

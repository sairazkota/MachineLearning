import quandl as Q
import pandas as pd
import math
import numpy as np
from sklearn import preprocesssing, cross_validation,svm
from sklearn.linearmodel import LinearRegression

'''Cross_Validations is basically used for separate/ shuffle the data to test like unbaised and
training and testing samples'''
'''
SVM: Support Vector Machine
'''

df=Q.get("WIKI/GOOGL", authtoken="cc21PMKA3pSvgH_2vraL")
#print(df.head())
#print("*"*150)
d_df=df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
d_df['LH_PCT']=(d_df['Adj. High'] - d_df['Adj. Close'])/d_df['Adj. Close']*100
d_df['PCT_Change']=(d_df['Adj. Close'] - d_df['Adj. Open'])/d_df['Adj. Open']*100
#print(d_df.head())
d_df=d_df[[ 'Adj. Close','PCT_Change','LH_PCT', 'Adj. Volume']]

forecast_col= 'Adj. Close'
d_df.fillna(-99999,inplace=True)
forecast_out=int(math.ceil(0.1*len(d_df)))
d_df['label']=d_df[forecast_col].shift(-forecast_out)
#print(d_df.head())
#print("*"*150)
d_df.dropna(inplace=True)
#print(d_df)

X=np.array(d_df.drop(['label'],1))
y=np.array(d_df['label'])
X=preprocessing.scale(X) #Scale the new values 
y=np.array(d_df['label'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#X_Train and Y_Train are fit for classifier
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)









"""
Please contact If you need any more details : sairaju.kota@adp.com/sairazkota@gmail.com
"""

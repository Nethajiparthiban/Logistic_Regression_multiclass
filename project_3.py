import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
data=pd.read_csv(r"D:\Git\Git-Projects\bmi.csv")
cat_col=['Gender']
other_col=['Height','Weight']
Y=data['Index']
X=data[cat_col+other_col]
#Encoding the data
ct=ColumnTransformer(transformers=[('cat_encode',OneHotEncoder(),cat_col)],remainder='passthrough')
x=ct.fit_transform(X)
#Training the data set
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=0)
#fitting to std scalar
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
#Fitting to the data set
scaler=LogisticRegression(max_iter=600)
scaler.fit(x_train,y_train)
score=scaler.score(x_test,y_test)
scaler.predict(x_test)
scaler.predict_proba(x_test)
scaler.coef_
scaler.intercept_
print(score*100)
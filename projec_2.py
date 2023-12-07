#Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#reading the data set
health=pd.read_csv(r"D:\Git\Git-Projects\fetal_health.csv")
#print(health.head())
#health.info()
#health.describe()
#print(health.isnull().sum())
X=health.iloc[:,:-1]
Y=health.iloc[:,-1]
#print(X)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
#print(x_test)
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
score=log_reg.score(x_train,y_train)
score2=log_reg.score(x_test,y_test)
y_pred=log_reg.predict(x_test)
prob=log_reg.predict_proba(x_test)
co_ef=log_reg.coef_
inter_cept=log_reg.intercept_
print(inter_cept)

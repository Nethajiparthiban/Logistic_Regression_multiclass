import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv(r"D:\Git\Git-Projects\titanic_train.csv")
#print(df)
#print(df.info())
#print(df.describe())
mean=df['Age'].mean()
df['Age']=df['Age'].fillna(mean)
#df['Embarked']=df['Embarked'].dropna(inplace=True)
#print(df.isnull().sum())
df.drop('Cabin',axis=1,inplace=True)
#print(df.isnull().sum())
x=df['Sex']
y=df['Survived']
#plt.plot(x,y)
#plt.show()
Y=df['Survived']
cat_col=['Sex']
ord_col=['Name','Ticket','Embarked']
other_col=['PassengerId','Pclass','Age','SibSp','Parch','Fare']
X=df[cat_col+ord_col+other_col]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
ct=ColumnTransformer(transformers=[('Cat_encode',OneHotEncoder(),cat_col),
                                ('Ord_encode',OrdinalEncoder(),ord_col)],remainder='passthrough')
x=ct.fit_transform(X)
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler
formaat=StandardScaler()
formaat.fit(x_train,y_train)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)
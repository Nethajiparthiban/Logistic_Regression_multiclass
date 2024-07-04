import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
iris=load_iris()
dir(iris)
iris.data[0]
data=iris.data
labels=iris.target
df_data=pd.DataFrame(data,columns=iris.feature_names)
df_labels=pd.DataFrame(labels,columns=['labels'])
df_combained=pd.concat([df_data,df_labels],axis=1)
#we have target values as 0,1,2
#which means 0-setosa,1-versicolor,2-virginica
df_combained['labels'].unique()
iris.target_names
#print(df_combained.head())
features=df_combained.drop(['labels'],axis='columns')
target=df_combained.labels
#print(target.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)
#print(len(x_train))
#print(y_test)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression(max_iter=10000)
reg.fit(x_train,y_train)
print(reg.score(x_test,y_test))
print(reg.predict([[5.8,2.8,5.1,2.4]]))
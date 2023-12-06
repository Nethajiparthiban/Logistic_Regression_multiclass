#Importing moduless
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#Loading the data set
df=pd.read_csv(r"D:\Git\Git-Projects\modifiedDigits4Classes.csv")
#print(df.head(5))
#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#visualize Each digit
pixel_colnames=df.columns[:-1]
#print(pixel_colnames)
image_values=df.loc[0,pixel_colnames].values
#print(image_values)
#print(image_values.reshape(8,8))
re_shape=image_values.reshape(8,8)
#plt.imshow(re_shape,cmap='rainbow')
#plt.show()
#plt.figure(figsize=(10,2))
#for index in range(0,5):
    #plt.subplot(1,5,1+index)
    #image_values=df.loc[index,pixel_colnames].values
    #image_label=df.loc[index,'label']
    #plt.imshow(re_shape,cmap='YlGn_r')
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(df[pixel_colnames],df['label'],test_size=0.25,random_state=0)
#fitting to the algoritham
clf=LogisticRegression(max_iter=1000)
clf.fit(x_train,y_train)
score1=clf.score(x_test,y_test)
score2=clf.score(x_train,y_train)
#print(score1)
#print(score2)
y_pred=clf.predict(x_test)
#print(y_pred)
#Probablity
prob=clf.predict_proba(x_test)
#print(prob)
#Co-efficent
coff=clf.coef_
#print(coff)
#Intercept
inter=clf.intercept_
#print(inter)
#score check
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import metrics
rscore=r2_score(y_test,y_pred)
rs=rscore*100
#print(rs.round(),'%')
import seaborn as sns
#score=clf.score(x_test,y_test)
#cm=metrics.confusion_matrix(y_test,clf.predict(x_test))
#plt.figure(figsize=(9,9))
#sns.heatmap(cm,annot=True,
            #fmt=".0f",
            #linewidths=0.5,
            #square=True,
            #cmap='Blues');
#plt.ylabel('Actual Label',fontsize=17);
#plt.xlabel('Predicted label',fontsize=17);
#plt.title('Accuracy score : {}'.format(score),size=17);
#plt.tick_params(labelsize=15)
#plt.show()

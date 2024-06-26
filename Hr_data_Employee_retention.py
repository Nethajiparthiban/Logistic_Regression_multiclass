import pandas as pd
df=pd.read_csv(r"C:\Users\Netha\Ananconda_onlineclass\Mission learning\ML from codebasics\Machine Learning\Logistic_regression\HR_comma_sep.csv")
#print(df.head())
#Data Exploration and visualisation
left=df[df.left==1]
#print(left.shape)
retained= df[df.left==0]
#print(retained.shape)
mean=df.groupby('left').mean()
#print(mean)
'''as we saw the above mean in the left column we might notice that the satisfection
level of the employee is poor when compare the retained also the number of ours 
 worked in the office is little higer and also promotion in the last 5yrs is very less'''
#now we create new df with required fileds so that makes df simpler.

subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
#print(subdf.head())
#as we saw the new df salary column has only text values so we use getdummies method

salary_dummies=pd.get_dummies(df.salary,prefix='salary')

#print(salary_dummies)
#lets concat the data
new_df=pd.concat([salary_dummies,subdf],axis='columns')
#print(new_df)
#lets drop the salary column in new df
new_df=new_df.drop(['salary'],axis='columns')
#print(new_df.columns)
x=new_df
y=df.left
#print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#print(len(x_train))
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=100000)
clf.fit(x_train,y_train)
#print(clf.predict(x_test))
print(clf.score(x_test,y_test))
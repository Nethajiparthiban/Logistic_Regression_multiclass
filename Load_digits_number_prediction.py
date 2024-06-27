import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
digits=load_digits()
print(dir(digits))
digits.images[0]
#the below loop will bring the 0-5 images
for i in range(5):
    plt.gray()
    plt.matshow(digits.images[i])
digits.target[0:5]
digits.target_names[0:5]
#lets split the data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)
print(len(x_test))
print(len(x_train))
#lets fit to the logistic regression model
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(max_iter=100000)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))
plt.matshow(digits.images[67])
print(digits.target[67])
print(clf.predict([digits.data[67]]))
#lets cross verify with confussion matrix
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
#lets visualize it
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
data=pd.read_csv(r"D:\Git\Git-Projects\INNHotelsGroup.csv")
#print(data)
x=data['market_segment_type']
y=data['avg_price_per_room']
#plt.scatter(x,y,color='blue')
#plt.show()
x=data['arrival_year']
y=data['avg_price_per_room']
#plt.scatter(x,y)
#plt.show()
cat_col=['market_segment_type']
ord_col=['Booking_ID','type_of_meal_plan','room_type_reserved']
other_col=['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
           'required_car_parking_space','lead_time','arrival_year','arrival_month','arrival_date',
           'repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled',
           'avg_price_per_room']
Y=data.iloc[:,-2]
#print(Y)
X=data[cat_col+ord_col+other_col]
#print(X.columns)
ct=ColumnTransformer(transformers=[('cat_encoder',OneHotEncoder(),cat_col),
                                   ('ord_encode',OrdinalEncoder(),ord_col)],remainder='passthrough')
x=ct.fit_transform(X)
#print(x)
#Training the data set.
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to the algoritham model.
modle=LogisticRegression()
modle.fit(x_train,y_train)
score=modle.score(x_test,y_test)
print(score)
import pandas as pd

df=pd.read_csv("insurance.csv")
x=df.iloc[:,0:1]
y=df['premium']
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=51)
from sklearn import linear_model
lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)
ans=lr.predict([[21]])
ans=lr.predict([[100]])
ans=lr.predict(x_test[0:1])
print(lr.score(x_test,y_test))
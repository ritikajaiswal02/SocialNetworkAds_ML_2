#Random Forest Classifier model

#import the lib
import pandas as pd
from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load the data
data = pd.read_csv("sna_aug25.csv")

#check for null data
print(data.isnull().sum())

#features and target
features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

#training and testing
x_train,x_test,y_train,y_test = train_test_split(features.values,target,random_state=6)

#model creation
model = RandomForestClassifier(n_estimators=10,random_state=6)
model.fit(x_train,y_train)

#model saving
f = open("model.pkl","wb")
dump(model,f)
f.close()

print("Model created")


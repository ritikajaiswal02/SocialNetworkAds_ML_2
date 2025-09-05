#random forest classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("sna_aug25.csv")
print(data)

features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

x_train,x_test,y_train,y_test = train_test_split(features.values,target,random_state=6)

model = RandomForestClassifier(n_estimators=10,random_state=6)
model.fit(x_train,y_train)

print("\n\n********Random Forest Classifier*******")

"""
print("\n********Confusion Matrix********\n")

cm = confusion_matrix(y_test,model.predict(x_test))
print(cm)
"""

print("\n********Classification Report********\n")
clf = classification_report(y_test,model.predict(x_test))
print(clf)


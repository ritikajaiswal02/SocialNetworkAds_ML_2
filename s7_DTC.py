#Decision Tree Classifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv("sna_aug25.csv")
print(data)

features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

x_train,x_test,y_train,y_test = train_test_split(features.values,target,random_state=10)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)

print("\n\n********Decision Tree Classifier********")

"""
print("\n********Confusion Matrix********\n")

cm = confusion_matrix(y_test,model.predict(x_test))
print(cm)
"""

print("\n********Classification Report********\n")
clf = classification_report(y_test,model.predict(x_test))
print(clf)

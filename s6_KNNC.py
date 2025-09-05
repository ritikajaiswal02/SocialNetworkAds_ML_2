import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("sna_aug25.csv")
print(data)

features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

mms = MinMaxScaler()
sfeatures = mms.fit_transform(features)

x_train,x_test,y_train,y_test = train_test_split(sfeatures,target,random_state=9)

k = int(len(data)**0.5)
if k % 2 == 0:
	k+=1

model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)

print("\n\n********KNN Classifier********")

"""
print("\n********Confusion Matrix********\n")

cm = confusion_matrix(y_test,model.predict(x_test))
print(cm)
"""

print("\n********Classification Report********\n")
clf = classification_report(y_test,model.predict(x_test))
print(clf)
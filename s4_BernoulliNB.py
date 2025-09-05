import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

data = pd.read_csv("sna_aug25.csv")
print(data)

features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

x_train,x_test,y_train,y_test = train_test_split(features.values,target,random_state=15,test_size=0.2)

model = BernoulliNB()
model.fit(x_train,y_train)

print("\n\n********Bernoulli NB********")

print("\n********Classification Report********\n")
clf = classification_report(y_test,model.predict(x_test),zero_division=True)
print(clf)



"""
For BernoulliNB: All features must be binary (0 or 1)

Not suitable for:
Continuous numerical features (like Age, EstimatedSalary, etc.)
Categorical or multi-class features without one-hot encoding or binarization
"""
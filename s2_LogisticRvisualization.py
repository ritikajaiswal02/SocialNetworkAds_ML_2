#visualizing the s shaped line

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("sna_aug25.csv")
print(data)

data.sort_values(by="EstimatedSalary",inplace=True)

print(data.isnull().sum())
#print(data)

x = data["EstimatedSalary"]
y = data["Purchased"]

features = data[["Age","EstimatedSalary"]]
target = data["Purchased"]

model = LogisticRegression()
model.fit(features.values,target)

plt.scatter(x,y,color="red")
plt.plot(data["EstimatedSalary"],model.predict(features),color="purple")
plt.xlabel("EstimatedSalary")
plt.ylabel("Purchased")
plt.title("Social Network Ads")
plt.show()
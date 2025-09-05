#visualizing the dataset

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sna_aug25.csv")
print(data)

x = data["EstimatedSalary"]
y = data["Purchased"]

plt.scatter(x,y,color="red")
plt.xlabel("EstimatedSalary")
plt.ylabel("Purchased")
plt.title("Social Network Ads")
plt.show()
# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load and explore customer data.

2.Determine Optimal Clusters: Use the Elbow Method to find the best number of clusters.

3.Apply K Means Clustering: Perform clustering on customer data.

4.Visualize Segmented Customers: Plot clustered data to visualize customer segments. 


## Program:
### Program to implement the K Means Clustering for Customer Segmentation.
### Developed by:Jwalamukhi S
### RegisterNumber:212223040079
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/black/Downloads/Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No of Cluster")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:

head():
![image](https://github.com/Jwalamukhi/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145953628/be9e6744-299b-49dd-864c-b17dbff1bd9d)

isnull().sum():
![image](https://github.com/Jwalamukhi/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145953628/ca412f44-0b92-4159-aac3-b2daa443008a)

graph:
![image](https://github.com/Jwalamukhi/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145953628/4c2280f9-cf33-472b-94a3-485bcb35e2f3)

y_pred:
![image](https://github.com/Jwalamukhi/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145953628/34cb5f1c-502d-4380-b097-af3f335951e3)

cluster:
![image](https://github.com/Jwalamukhi/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/145953628/d2921ae1-b8a1-4c6e-8ae9-e9b1c5e2f859)








## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

import sklearn
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import iris toy data set 
from sklearn.datasets import load_iris
iris = load_iris()

irisdata = pd.DataFrame(iris.data,columns=iris.feature_names)
print(irisdata.head())

# Calculate each SSE and plot
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(irisdata)
    irisdata["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#We can see in the plot that 3 is the optimal number of clusters or iris dataset, which is indeed correct.
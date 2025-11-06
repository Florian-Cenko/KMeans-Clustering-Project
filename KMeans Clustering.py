import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

# K Means Clustering Project
#   For this project we will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.

# 1.First I get and print the data
college_data = pd.read_csv('College_Data',index_col=0)
print(college_data.head())
print(college_data.info())
print(college_data.describe())

# 2. Now I do Exploratory Data Analysis
# a.Create a scatterplot of Grad.Rate versus Room.Board where
# the points are colored(hue='Private')by the Private column.
sns.scatterplot(college_data,x='Room.Board',y='Grad.Rate',hue='Private')
plt.show()

# b. Same for  F.Undergrad versus Outstate
sns.scatterplot(college_data,x='Outstate',y='F.Undergrad',hue='Private')
plt.show()


# c. ** Create a stacked histogram showing Out of State Tuition based on the Private column.
# Try doing this using [sns.FacetGrid]
g = sns.FacetGrid(data=college_data,hue='Private')
g.map(sns.histplot,'Outstate')
plt.show()


# d. **Create a similar histogram for the Grad.Rate column.**
g = sns.FacetGrid(data=college_data,hue='Private')
g.map(sns.histplot,'Grad.Rate')
plt.show()

grad_Data = college_data[college_data['Grad.Rate']>100]
print(grad_Data.index[0])

#college_data['Grad.Rate']['Cazenovia College'] = 100


## 3. K Means Cluster Creation
# a. ** Create an instance of a K Means model with 2 clusters.**
kmeans = KMeans(n_clusters=2)

# b. **Fit the model to all the data except for the Private label.**
kmeans.fit(college_data.drop('Private',axis=1))

# Cluster Centres
print(kmeans.cluster_centers_)

## Evaluation of KMeans
#** Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**
def func(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

# SOS απο συναρτηση με apply μεταφερω τις τιμες και επειτα χρηση confussion_matrix and classification_report
college_data['Cluster'] = college_data['Private'].apply(func)

print(confusion_matrix(college_data['Cluster'],kmeans.labels_))
print("\n")

print(classification_report(college_data['Cluster'],kmeans.labels_))
print("\n")

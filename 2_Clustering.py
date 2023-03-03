import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


### Load Data ###
# Read csv function #
def read_csv(path):
    dirs = os.listdir(path)
    
    for file in dirs:
        if ".csv" in file:
            print(file)
            raw_df = pd.read_csv(path + file) 
    
    return raw_df

# Data preprocessing #
path = 'C:/Users/ilove/Retailing/Temp/英業達/'
raw_df = read_csv(path)
for i in raw_df.columns:
    if i not in ["name", "party", "state"]:
        raw_df = raw_df.rename(columns={i: "col"+i})

### K-means Clustering ###
## Data Preprocessing ##
k_df = raw_df.drop(["name", "party", "state"], axis=1)

# Change value if value=0.5 to columns mode value(0 or 1) #
for i in k_df.columns:
    k_df[i] = k_df[i].apply(lambda x: k_df[i].mode()[0] if x == 0.5 else x)

## Define new customer cluster #
# Choose which column to define new customer #
value_count = pd.DataFrame()
for i in k_df.columns:
    temp = k_df[i].value_counts()
    value_count = pd.concat([value_count, temp], axis=1)
# col4 = 0 define to be new customer
k_df["value_label"] = k_df["col4"].apply(lambda x: "new" if x == 0 else np.nan)

## RFM model ##
# Set weight for columns #
for i in k_df.drop(["col4", "value_label"], axis=1).columns:
    weight = np.random.uniform(low=0.1, high=10)
    k_df[i] = k_df[i] * weight
# Define recency, frequency and monetary #
r = ["col" + f"{i}" for i in (1,2,3,5)]
f = ["col" + f"{i}" for i in range(6, 11)]
m = ["col" + f"{i}" for i in range(11, 16)]
k_df["r"] = k_df[r].sum(axis=1)
k_df["f"] = k_df[f].sum(axis=1)
k_df["m"] = k_df[m].sum(axis=1)

rfm_df = k_df[["r", "f", "m", "value_label"]]
rfm_df_new = rfm_df[rfm_df["value_label"] == "new"]
rfm_df = rfm_df[rfm_df["value_label"] != "new"]

## K-means model ##
# K-means with best num of clusters by elbow method & silhouette analysis #
wss = []
silhouette_avg = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans_fit = kmeans.fit(rfm_df[["r", "f", "m"]])
    wss.append(kmeans.inertia_)
    if i != 1:
        silhouette_avg.append(silhouette_score(rfm_df[["r", "f", "m"]], kmeans_fit.labels_))

plt.plot(range(1,11), wss, marker='o')
plt.title('Elbow graph')
plt.xlabel('Cluster number')
plt.ylabel('WSS')
# k=4
plt.plot(range(2,11), silhouette_avg, marker='o', color='r')
plt.title('Silhouette graph')
plt.xlabel('Cluster number')
plt.ylabel('Silhouette coefficient values')
# k=9

# k-means model #
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=0)
rfm_df['cluster_label'] = kmeans.fit_predict(rfm_df[["r", "f", "m"]])

center = pd.DataFrame(kmeans.cluster_centers_, columns=["R", "F", "M"])
center = pd.concat([center, center.apply(sum, axis=1)], axis=1).reset_index().rename(columns={"index": "cluster_label", 0: "SUM"})

# Value_label algorithm #
def value_label(col):
    if col == center["SUM"].max():
        return "high"
    elif col == center["SUM"].min():
        return "lost"
    else:
        return "other"
def loyal(df):
    label = df[0]
    fr_m = df[1]
    if label == "other":
        if fr_m == center["FM-R"].max():
            return "sleep"
        else:
            return "loyal"
    else:
        return label
center["label"] = center["SUM"].apply(value_label)
center["FM-R"] = center["F"] + center["M"] - center["R"]
center["label"] = center[["label", "FM-R"]].apply(loyal, axis=1)

rfm_df = pd.merge(rfm_df, center[["cluster_label", "label"]], how="left", on="cluster_label")
rfm_df["value_label"] = rfm_df["label"]
rfm_df = pd.concat([rfm_df.drop(["cluster_label", "label"], axis=1), rfm_df_new])
rfm_df["value_label"].value_counts()

# Monetary & Recency Plot #
sns.scatterplot(data=rfm_df, x="m", y="r", hue="value_label", palette="tab10")



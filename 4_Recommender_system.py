import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity


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


### Recommender system ###
## Data Preprocessing ##
rs_df = raw_df.drop(["party", "state"], axis=1)

# Change value if value=0.5 to columns mode value(0 or 1) #
for i in rs_df.columns:
    if i != "name":
        rs_df[i] = rs_df[i].apply(lambda x: rs_df[i].mode()[0] if x == 0.5 else x)

# Rename columns #
for i in range(1, 16):
    if i != "name":
        rs_df.rename(columns={"col"+str(i): "product"+str(i)}, inplace=True)


## Collaborative filtering ##
# Item-item similarity matrix by cosine similarity method #
rs_df.index = rs_df["name"]
matrix = rs_df.drop(["name"], axis=1).T
matrix_similarity = cosine_similarity(matrix)
np.fill_diagonal(matrix_similarity, 0)
matrix_similarity = pd.DataFrame(matrix_similarity, columns=matrix.index).set_index(matrix.index).reset_index()

# similarity matrix plot #
plt.figure(figsize=(14, 8))
sns.heatmap(matrix_similarity.set_index("index"), annot=True, linewidth=.5, cmap="autumn")

# Each user's recommend product by total scoring #
rec_df = pd.DataFrame()
for j in rs_df["name"]:
    product = []
    row = rs_df[rs_df["name"] == j]
    for i in row.columns:
        if row.loc[j, i] == 1:
            # each user by which product #
            product.append(i) 
            # total score from similarity matrix #
            score = pd.DataFrame(matrix_similarity[matrix_similarity["index"].isin(product)].sum()).reset_index()
            score = score[score["index"] != "index"].sort_values(by=[0], ascending=False).rename(columns={"index": "rec_prod", 0: "score"})
            # top 5 recommend products for each user #
            top5rec = pd.DataFrame(score["rec_prod"].head(5)).T.reset_index(drop=True)
            top5rec.columns = ["re1", "rec2", "rec3", "rec4", "rec5"]
            top5rec.insert(0, "name", j)
    rec_df = pd.concat([rec_df, top5rec])



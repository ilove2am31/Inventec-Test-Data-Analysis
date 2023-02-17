import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


### Load Data & EDA ###
## Load Data ##
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

# Check if duplicate names exits #
duplicate_name = raw_df.groupby(["name"])["name"].count()
duplicate_name[duplicate_name > 1] #0


## EDA ##
# Party Bar chart #
party = raw_df.groupby(["party"])["party"].count().reset_index(name='counts')
sns.set_style("whitegrid")
sns.barplot(data=party, x="party", y="counts", palette="deep", ci=None)

# State Pie chart #
state = raw_df.groupby(["state"])["state"].count().reset_index(name='counts')
pie, ax = plt.subplots(figsize=[12, 12])
labels = state["state"]
plt.pie(x=state["counts"], autopct="%.1f%%", labels=labels, shadow=True, startangle=90, textprops={"fontsize":20})

# Overview of cols1~15 #
cols_count = pd.DataFrame()
for i in raw_df.columns:
    if i not in ["name", "party", "state"]:
        temp = raw_df[i].value_counts()
        cols_count = pd.concat([cols_count, temp], axis=1)

# Top 5 value=1 cols #
top5one = cols_count.T.sort_values(1.0, ascending=False).head(5).reset_index().rename(columns={"index": "col_name", 1.0: "value=1"})
sns.barplot(data=top5one, x="col_name", y="value=1", palette="autumn", ci=None)
plt.ylim(0, 100)
# Top 5 value=0 cols #
top5zero = cols_count.T.sort_values(0.0, ascending=False).head(5).reset_index().rename(columns={"index": "col_name", 0.0: "value=0"})
sns.barplot(data=top5zero, x="col_name", y="value=0", palette="winter", ci=None)
plt.ylim(0, 100)

# value=0.5 Displot #
sns.set_style("white")
dis = cols_count.T.rename(columns={0.5: "value=0.5 counts"})
sns.displot(data=dis, x="value=0.5 counts", kind="kde")

# Correlation Matrix #
corr = raw_df.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr, annot=True, linewidth=.5)

# Party with label(col15) Bar chart #
party_label = raw_df.loc[raw_df["col15"] != 0.5, ["party", "col15"]]
party_label.columns = ["party", "label"]
party_label = party_label.groupby(["party", "label"])["label"].count().reset_index(name='counts')
sns.set_style("white")
sns.barplot(data=party_label, x="party", y="counts", hue="label", palette="husl", ci=None)



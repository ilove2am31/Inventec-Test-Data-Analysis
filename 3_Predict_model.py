import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm  import LGBMClassifier


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


### Modeling for predict col15
## Data Preprocessing ##
# drop useless columns #
df = raw_df.drop(["name", "state"], axis=1)

# Change value if value=0.5 to columns mode value(0 or 1) #
for i in df.columns:
    if i != "party":
        df[i] = df[i].apply(lambda x: df[i].mode()[0] if x == 0.5 else x)

# Dummy variables for party column #
df = pd.get_dummies(data=df, columns=["party"], drop_first=True)

# Train test split #
# (Define col15 to be label)
X = df.drop(["col15"], axis=1)
y = df["col15"]
#y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


## Model ##
model_compare = pd.DataFrame(columns=["model", "accuracy", "f1_score", "roc_auc", "run_time"])

# 1. Logistic regression with grid search #
# Grid Search #
parameters = {"penalty": ["l1", "l2"], 
              "C": [1, 10, 100, 1000]}
grid_lr = GridSearchCV(estimator = LogisticRegression(),  
                       param_grid = parameters,
                       scoring = 'accuracy',
                       cv = 5,
                       verbose=0)
grid_lr.fit(X_train, y_train)
print("best parameters: ", grid_lr.best_params_, "\naccuracy: ", grid_lr.best_score_)

# Logistic regression model #
logreg = LogisticRegression(C=10, penalty='l2')
t = time.time()
logreg.fit(X_train, y_train)
ts = time.time() - t
y_pred = logreg.predict(X_test)
print("Accuracy: ", logreg.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

logreg_score = pd.DataFrame([["Logistic Regression", logreg.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, logreg_score])


# 2. K-nearest neighbors(KNN) with best k #
# Best k #
neig = np.arange(1, 25)
train_acc = []
test_acc = []
# Loop over different values of k #
for i, k in enumerate(range(1, 21)):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc.append(knn.score(X_train, y_train))
    test_acc.append(knn.score(X_test, y_test))
plt.figure(figsize=[12, 6])
plt.plot(range(1, 21), test_acc, label = 'Testing Accuracy')
plt.plot(range(1, 21), train_acc, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(range(1, 21))

# KNN model
knn = KNeighborsClassifier(n_neighbors=1)
t = time.time()
knn.fit(X_train, y_train)
ts = time.time() - t
y_pred = knn.predict(X_test)
print("Accuracy: ", knn.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

knn_score = pd.DataFrame([["KNN", logreg.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, knn_score])


# 3. Support vector machine(SVM) with grid search #
# Grid Search #
parameters = {"C": [0.01, 0.1, 0.5, 1, 10],
              "kernel": ["linear", "poly", "rbf"],
              "gamma": [0.01, 0.1, 0.5, 1, 10]}
svc_grid = GridSearchCV(estimator = SVC(), 
                        param_grid = parameters,
                        scoring = "accuracy",
                        cv = 5,
                        verbose = 0)
svc_grid.fit(X_train, y_train)
print("best parameters: ", svc_grid.best_params_, "\naccuracy: ", svc_grid.best_score_)

# SVM model #
svc = SVC(C=0.5, kernel="rbf", gamma=0.5)
t = time.time()
svc.fit(X_train, y_train)
ts = time.time() - t
y_pred = svc.predict(X_test)
print("Accuracy: ", svc.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

svc_score = pd.DataFrame([["SVM", svc.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, svc_score])


# 4. Decision tree with grid search #
# Grid Search #
parameters = {"criterion": ["gini", "entropy"],
              "max_depth": [i for i in range(1, 21)],
              "min_samples_split": [2, 3, 4],
              "min_samples_leaf": [i for i in range(1, 11)]}
dt_grid = GridSearchCV(estimator = DecisionTreeClassifier(), 
                        param_grid = parameters,
                        scoring = "accuracy",
                        cv = 5,
                        verbose = 0)
dt_grid.fit(X_train, y_train)
print("best parameters: ", dt_grid.best_params_, "\naccuracy: ", dt_grid.best_score_)

# Decision tree model #
dt = DecisionTreeClassifier(criterion= "gini", max_depth=3, min_samples_leaf=1, min_samples_split=2)
t = time.time()
dt.fit(X_train, y_train)
ts = time.time() - t
y_pred = dt.predict(X_test)
print("Accuracy: ", dt.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

dt_score = pd.DataFrame([["Decision Tree", dt.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, dt_score])


# 5. Random Forest with random search #
# Grid Search #
parameters = {"n_estimators": [int(i) for i in np.linspace(start=100, stop=1000, num=10)],
              "max_features": ["sqrt", "log2", None],
              "max_depth": [int(i) for i in range(1, 11)],
              "min_samples_split": [2, 3, 4],
              "min_samples_leaf": [1, 3, 5],
              "bootstrap": [True, False]}
rf_grid = RandomizedSearchCV(estimator = RandomForestClassifier(), 
                             param_distributions = parameters,
                             scoring = "accuracy",
                             n_iter = 200, 
                             cv = 5,
                             verbose = 0)
rf_grid.fit(X_train, y_train)
print("best parameters: ", rf_grid.best_params_, "\naccuracy: ", rf_grid.best_score_)

# Random forest model #
rf = RandomForestClassifier(n_estimators=700, min_samples_split=4, min_samples_leaf=1, max_features=None, max_depth=3, bootstrap=True)
t = time.time()
rf.fit(X_train, y_train)
ts = time.time() - t
y_pred = rf.predict(X_test)
print("Accuracy: ", rf.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

rf_score = pd.DataFrame([["Random Forest", rf.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, rf_score])

# Feature importance plot #
rf_feature_imp = pd.DataFrame([X_train.columns, rf.feature_importances_]).T.sort_values(by=1)
rf_feature_imp.columns = ["col_name", "importance"]
plt.barh(rf_feature_imp["col_name"], rf_feature_imp["importance"])
plt.xlabel("Feature Importance")


# 6. XGBoost with random search #
parameters = {"n_estimators": [int(i) for i in np.linspace(start=100, stop=1000, num=10)],
              "eta": [0.01, 0.05, 0.1, 0.2, 0.3],
              "max_depth": [int(i) for i in range(1, 11)],
              "subsample": [0.2, 0.6, 1],
              "colsample_bytree": [0.2, 0.6, 1],
              "reg_alpha": [i for i in np.linspace(0, 1, 11)],
              "reg_lambda": [i for i in range(1, 11)]}
xgb_grid = RandomizedSearchCV(estimator = XGBClassifier(), 
                              param_distributions = parameters,
                              scoring = "accuracy",
                              n_iter = 200, 
                              cv = 5,
                              verbose = 0)
xgb_grid.fit(X_train, y_train, 
             early_stopping_rounds = 5,
             eval_set = [(X_test, y_test)],
             verbose = 0)
print("best parameters: ", xgb_grid.best_params_, "\naccuracy: ", xgb_grid.best_score_)

# Xgboost model #
xgb = XGBClassifier(n_estimators=500, subsample=1, reg_lambda=7, reg_alpha=0.6, max_depth=2, eta=0.2, colsample_bytree=1)
t = time.time()
xgb.fit(X_train, y_train)
ts = time.time() - t
y_pred = xgb.predict(X_test)
print("Accuracy: ", xgb.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

xgb_score = pd.DataFrame([["Xgboost Classifier", xgb.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, xgb_score])

# Feature importance plot #
xgb_feature_imp = pd.DataFrame([X_train.columns, xgb.feature_importances_]).T.sort_values(by=1)
xgb_feature_imp.columns = ["col_name", "importance"]
plt.barh(xgb_feature_imp["col_name"], xgb_feature_imp["importance"])
plt.xlabel("Feature Importance")


# 7. LightGBM with random search #
parameters = {"n_estimators": [int(i) for i in np.linspace(start=100, stop=1000, num=10)],
              "num_leaves": [10, 30, 50],
              "max_depth": [int(i) for i in range(1, 11)],
              "learning_rate": [0.1, 0.3, 0.7],
              "subsample": [0.2, 0.6, 1],
              "colsample_bytree": [0.2, 0.6, 1],
              "reg_alpha": [i for i in np.linspace(0, 1, 11)],
              "reg_lambda": [i for i in range(1, 11)]}
lgbm_grid = RandomizedSearchCV(estimator = LGBMClassifier(), 
                               param_distributions = parameters,
                               scoring = "accuracy",
                               n_iter = 200, 
                               cv = 5,
                               verbose = 0)
lgbm_grid.fit(X_train, y_train, 
              early_stopping_rounds = 5,
              eval_set = [(X_test, y_test)],
              verbose = 0)
print("best parameters: ", lgbm_grid.best_params_, "\naccuracy: ", lgbm_grid.best_score_)

# Lgbm model #
lgbm = LGBMClassifier(n_estimators=100, subsample=1, num_leaves=30, reg_lambda=10, reg_alpha=0.2, max_depth=5, learning_rate=0.7, colsample_bytree=1)
t = time.time()
lgbm.fit(X_train, y_train)
ts = time.time() - t
y_pred = lgbm.predict(X_test)
print("Accuracy: ", lgbm.score(X_test, y_test))
print("F1 score: ", f1_score(y_test, y_pred))
print("ROC AUC: ", roc_auc_score(y_test, y_pred))

lgbm_score = pd.DataFrame([["LightGBM Classifier", lgbm.score(X_test, y_test), f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred), ts]], columns=model_compare.columns)
model_compare = pd.concat([model_compare, lgbm_score])

# Feature importance plot #
lgbm_feature_imp = pd.DataFrame([X_train.columns, lgbm.feature_importances_]).T.sort_values(by=1)
lgbm_feature_imp.columns = ["col_name", "importance"]
plt.barh(lgbm_feature_imp["col_name"], lgbm_feature_imp["importance"])
plt.xlabel("Feature Importance")

confusion_matrix(y_test, y_pred)


## Model Compare ##
# Peformance #
for i,j in zip(["accuracy", "f1_score", "roc_auc"], ["r", "b", "y"]):
    plt.plot(model_compare["model"], model_compare[i], label=i, color=j, marker=".", markersize=5)
plt.xticks(rotation=270)
plt.legend()

# Run time #
plt.plot(model_compare["model"], model_compare["run_time"], label="run_time", color="m", marker=".", markersize=5)
plt.xticks(rotation=270)
plt.legend()



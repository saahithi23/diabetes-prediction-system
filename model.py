import numpy as np
import pandas as pd
from time import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Importing the dataset and dropping duplicates to fix data leakage
dataset = pd.read_csv('diabetes3.csv').drop_duplicates()
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 72)

def evaluate_model(model_name, y_test, y_pred, start, end):
    print(f"--- {model_name} ---")
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"F-measure: {f1_score(y_test, y_pred)*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")

# 1. Random Forest
start_rf = time()
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=8, criterion='entropy', random_state=72)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
end_rf = time()
evaluate_model("Random Forest", y_test, rf_pred, start_rf, end_rf)
pickle.dump(rf_classifier, open("rf_model.pkl", "wb"))

# 2. Logistic Regression
start_lr = time()
lr_classifier = LogisticRegression(max_iter=1000, random_state=72)
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)
end_lr = time()
evaluate_model("Logistic Regression", y_test, lr_pred, start_lr, end_lr)
pickle.dump(lr_classifier, open("lr_model.pkl", "wb"))

# 3. Decision Tree
start_dt = time()
dt_classifier = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=72)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
end_dt = time()
evaluate_model("Decision Tree", y_test, dt_pred, start_dt, end_dt)
pickle.dump(dt_classifier, open("dt_model.pkl", "wb"))

print("All models trained and saved successfully.")

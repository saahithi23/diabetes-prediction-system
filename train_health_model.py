import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time

print("Loading dataset...")
df = pd.read_csv('health_dataset/diabetes_binary_health_indicators_BRFSS2015.csv')

# Select the most important and interpretable features for the UI
# 1. HighBP (0/1)
# 2. HighChol (0/1)
# 3. BMI (continuous)
# 4. Smoker (0/1)
# 5. HeartDiseaseorAttack (0/1)
# 6. PhysActivity (0/1)
# 7. GenHlth (1-5)
# 8. Age (1-13)
selected_features = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 
    'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth', 'Age'
]

X = df[selected_features]
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest model on Health Indicators dataset...")
start = time.time()
rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
end = time.time()

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Training completed in {end - start:.2f} seconds")
print(f"Accuracy with 8 selected features: {acc * 100:.2f}%")

print("Saving model to rf_health_model.pkl...")
with open("rf_health_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model saved successfully.")

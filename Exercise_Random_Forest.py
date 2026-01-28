import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("Exercise_Random_Forest.csv")

le_brand = LabelEncoder()
le_motor = LabelEncoder()
df["Brand_enc"] = le_brand.fit_transform(df["Brand"])
df["Motor_enc"] = le_motor.fit_transform(df["Motor_Type"])

X = df[["Brand_enc","Year","Battery_kWh","Range_miles","Top_Speed_mph","Acceleration_0_60","Motor_enc"]]
y = df["Fast_Charger_Compatible"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)

plt.figure(figsize=(8,5))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

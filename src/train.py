import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load processed data
df = pd.read_csv("data/processed/processed_train.csv")

# Feature-Target Split
x = df.drop(columns=['booking_status'])
y = df['booking_status']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.4f}")

# Saving Model
pickle.dump(model, open("models/model.pkl", "wb"))
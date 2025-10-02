# Necessary imports
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load example dataset (Iris)
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define outputs directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the trained model
model_path = os.path.join(output_dir, "irissclassifier_model.joblib")
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")

# Optional: Later, you can load the model like this:
# loaded_model = joblib.load(model_path)
# predictions = loaded_model.predict(X_test)
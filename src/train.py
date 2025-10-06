
import argparse
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to set reproducibility
def set_seed(seed):
    np.random.seed(seed)

# Function to train and evaluate the classifier
def train_model(output_path, test_size, seed, n_estimators):
    # Set reproducibility
    set_seed(seed)

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(clf, output_path)
    print(f"Model saved to {output_path}")

# CLI argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--output", type=str, default="iris_model.joblib",
                        help="Output path to save the trained model")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of dataset used for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the Random Forest")

    args = parser.parse_args()
    train_model(args.output, args.test_size, args.seed, args.n_estimators)

    import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

# Create a ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Iris Classifier")

# Define the output folder path
output_folder = r"C:\Users\DELL\Desktop\ris-classifier1\outputs"

# Ensure the folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the full path for the saved plot
output_path = os.path.join(output_folder, "confusion_matrix.png")

# Save the plot to the specified path
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the saved path
print(f"Confusion matrix saved at: {output_path}")

# Optionally display the plot
plt.show()

import joblib

joblib.dump(output_folder, "random_forest_model.pkl")
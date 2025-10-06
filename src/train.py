
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
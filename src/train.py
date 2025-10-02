from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(test_size, random_state):
    """
    Main function to train and evaluate a machine learning model.
    
    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    # Load the Iris dataset
    print("Loading dataset...")
    data = load_iris()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    print(f"Splitting dataset with test_size={test_size} and random_state={random_state}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train a Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to use for testing (default: 0.2).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(test_size=args.test_size, random_state=args.random_state)
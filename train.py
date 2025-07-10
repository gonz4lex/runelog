from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from tracker import Tracker


def main():
    # Initialize the Tracker
    tracker = Tracker()

    # Create (or retrieve) an experiment
    experiment_id = tracker.get_or_create_experiment("Example")

    # Define model and hyperparameters
    params = {
        "C": 1.0,
        "solver": "liblinear",
        "random_state": 0
    }
    model = LogisticRegression(**params)

    # Prepare the data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    with tracker.start_run(experiment_id=experiment_id) as run_id:
        print(f"Started Run: {run_id}")

        # Log hyperparameters
        tracker.log_param("C", params["C"])
        tracker.log_param("solver", params["solver"])
        tracker.log_param("dataset_shape", list(X.shape))

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model and log metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        tracker.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy:.4f}")

        # Log the trained model as an artifact
        tracker.log_model(model, "model.pkl")

    print("\nRun finished.")

    # Load and display the results for the entire experiment
    print("\n--- Experiment Results ---\n")
    results_df = tracker.load_results(experiment_id)
    print(results_df)

if __name__ == "__main__":
    main()
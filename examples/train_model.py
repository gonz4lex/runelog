from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from runelog import get_tracker


def main():
    # Initialize the tracker
    tracker = get_tracker()

    # Create (or retrieve) an experiment
    experiment_id = tracker.get_or_create_experiment("example-train-model")

    # Define model and hyperparameters
    params = {"C": 1.0, "solver": "liblinear", "random_state": 0}
    model = LogisticRegression(**params)

    # Prepare the data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    run_id = None
    model_artifact_name = "model.pkl"

    with tracker.start_run(experiment_id=experiment_id) as active_run_id:
        run_id = active_run_id
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

    if run_id:
        # Register a model (with tags)
        registered_model_name = "customer-churn-model"
        tags = {"status": "candidate", "scope": "global"}
        version = tracker.register_model(
            run_id, model_artifact_name, registered_model_name, tags=tags
        )

        # Load a registered model
        loaded_model = tracker.load_registered_model(registered_model_name)
        print("Model loaded successfully:", loaded_model)

        # Add or change tags later
        tracker.add_model_tags(registered_model_name, version, {"status": "winner"})

        # Retrieve and print tags
        model_tags = tracker.get_model_tags(registered_model_name, version)
        print(f"\nTags for '{registered_model_name}' v{version}: {model_tags}")

    # Load and display the results for the entire experiment
    print("\n--- Experiment Results ---\n")
    results_df = tracker.load_results(experiment_id)
    print(results_df)


if __name__ == "__main__":
    main()

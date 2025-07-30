# Quickstart
## You First RuneLog Experiment

Welcome to Runelog! This guide will walk you through the entire process of tracking a simple machine learning model.

### Step 1: Installation
If you haven't already, install the runelog library from PyPI:

```bash
pip install runelog
```

### Step 2: Create a Training Script

Create a new Python file (e.g., quickstart.py) and paste the following code into it. This script trains a simple classification model and uses runelog to track its parameters and performance.

```python
from runelog import get_tracker
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 1. Initialize the tracker
# This is the main entry point to the RuneLog library.
tracker = get_tracker()

# 2. Start a new run within an experiment
# If "quickstart-example" doesn't exist, it will be created automatically.
with tracker.start_run(experiment_name="quickstart-example"):
    
    # Define and log the model's hyperparameters
    params = {"solver": "liblinear", "C": 0.5}
    tracker.log_parameter("solver", params["solver"])
    tracker.log_parameter("C", params["C"])
    print("Logged parameters:", params)

    # Your model training logic
    X, y = make_classification(n_samples=100, random_state=0)
    model = LogisticRegression(**params).fit(X, y)

    # Log the model's performance metric
    accuracy = accuracy_score(y, model.predict(X))
    tracker.log_metric("accuracy", accuracy)
    print(f"Logged accuracy: {accuracy:.4f}")

    # Log the trained model file as an artifact
    tracker.log_model(model, "logreg.pkl")
    print("Logged model: logreg.pkl")

print("\nRun finished!")
```

### Step 3: Run the Script

Execute the script from your terminal:

```bash
python quickstart.py
```

You will see the logged parameters and metrics printed to your console. In the background, **RuneLog** has saved all of this information into a new `.mlruns` directory.

### Step 4: Review Results in the UI

Launch the Streamlit UI with the following command:

```bash
# Make sure you are in the same root directory where your .mlruns folder was created
streamlit run app/main.py
```

Your browser will now open the Experiment Explorer. Select the "Quickstart Example" experiment, and you will see the run you just completed, along with its parameters and metrics, in a clean, interactive table.
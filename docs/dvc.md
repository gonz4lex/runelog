# DVC Integration Guide

RuneLog provides a lightweight integration with [DVC (Data Version Control)](https://dvc.org/) to create a robust, reproducible link between your experiment runs and the exact version of the data you used.

### Example Workflow

#### 1. Version Your Data with DVC

Before running your experiment, use DVC to track your dataset. This command saves a "fingerprint" of your data in a small `.dvc` file, which you can commit to Git.

```bash
dvc add data/iris.csv

git add data/iris.csv.dvc
git commit -m "data: track v1 of iris dataset"
```

#### 2. Log the DVC Input in Your Script

In your training script, use the `tracker.log_dvc_input()` method. It will find the `.dvc` file, read the unique data hash, and log it with your run.

```python
from runelog import get_tracker

tracker = get_tracker()

with tracker.start_run(experiment_name="model-with-dvc"):
    # This creates a permanent link to the data version
    tracker.log_dvc_input("data/iris.csv", name="training_set")
    
    # ... rest of your training and logging code
```

#### 3. View the Results

Your run is now permanently linked to the specific version of your dataset. In the RuneLog UI, you can inspect the run to see the exact MD5 hash of the data that was used, ensuring full traceability.
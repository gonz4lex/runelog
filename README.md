# Runelog
## Lightweight ML Tracker

A simple, file-based Python library for tracking machine learning experiments, inspired by MLflow.

**Current Status**:  In active development. The core API is functional but subject to change.

The name *Runelog* is a play on words. It evokes the common `run.log()` command used to log an experiment, while also treating your powerful, and sometimes mysterious, models as modern-day mystical writings: a "log of runes".

---

##  Why Runelog?

- **Zero-Overhead Setup** – start tracking runs in a single line
- **Local-First, Lightweight** – perfect for solo devs or small teams
- **Portable & Transparent** – data is stored in simple folders/files

---

##  Installation

1. **Clone the repository**:

```bash
git clone https://github.com/YOUR_USERNAME/runelog.git
cd runelog
```
2. **Create and activate a virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. **Install dependencies**:

```bash
pip install -r requirements.txt
```
#### Quickstart Example
Use the `Tracker` class to log your model training process.

```python
from runelog import Tracker
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tracker = Tracker()
experiment_id = tracker.create_experiment("Example")

with tracker.start_run(experiment_id=experiment_id):
    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    tracker.log_metric("accuracy", acc)
    tracker.log_model(model, "logistic_model.pkl")
```
#### Usage Examples
You can find example scripts in the `examples/ directory`:

`train_model.py`
Full pipeline example with:
* logging parameters and metrics
* saving and registering models
* tagging and retrieving models

Run it:

```bash
python examples/train_model.py
```

`minimal_tracking.py`
Minimal working example with only metric logging.

Run it:

```bash
python examples/minimal_tracking.py
```
---
#### Planned Features
- Core tracking API (experiments, runs, params, metrics)
- Model artifact saving and registry
- Tagging and versioning
- Streamlit UI to view and compare runs
- Full test coverage
- Enhanced CLI
---


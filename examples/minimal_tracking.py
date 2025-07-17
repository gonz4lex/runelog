from runelog import Tracker
import time

def main():
    tracker = Tracker()
    experiment_id = tracker.get_or_create_experiment("minimal-example")

    with tracker.start_run(experiment_id=experiment_id):
        tracker.log_param("experiment_type", "minimal")
        for epoch in range(3):
            tracker.log_metric("accuracy", 0.75 + epoch * 0.05)
            time.sleep(0.5)

    print("Minimal tracking complete.")

if __name__ == "__main__":
    main()

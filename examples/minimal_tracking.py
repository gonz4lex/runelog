from runelog import get_tracker
import time

def main():
    tracker = get_tracker()
    experiment_name = "example-minimal-tracking"

    with tracker.start_run(experiment_name=experiment_name):
        tracker.log_param("experiment_type", "minimal")
        for epoch in range(3):
            tracker.log_metric("accuracy", 0.75 + epoch * 0.05)
            time.sleep(0.5)

    print("Minimal tracking complete.")

if __name__ == "__main__":
    main()

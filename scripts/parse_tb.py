import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def read_tb_logs():
    tb_runs_dir = "logs/tb_runs"
    # Find latest run
    all_runs = glob.glob(os.path.join(tb_runs_dir, "sacred_atla_*"))
    all_runs.sort(key=os.path.getmtime)
    if not all_runs:
        print("No runs found.")
        return
    latest_run = all_runs[-1]
    
    # Find latest event file
    event_files = glob.glob(os.path.join(latest_run, "events.out.tfevents.*"))
    if not event_files:
        print("No event files found.")
        return
    latest_event = event_files[0]
    
    ea = event_accumulator.EventAccumulator(latest_event)
    ea.Reload()
    
    print(f"Metrics from {latest_run}:")
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        latest_event = events[-1]
        print(f"  {tag}: {latest_event.value:.4f} (Step {latest_event.step})")

if __name__ == "__main__":
    read_tb_logs()

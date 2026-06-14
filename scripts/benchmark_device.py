import time
import subprocess
import sys

def run_benchmark(device: str, episodes: int = 2):
    start = time.time()
    cmd = [
        sys.executable,
        "scripts/train_sacred.py",
        "--episodes", str(episodes),
        "--device", device,
        "--switch-every", "5",
        "--batch-size", "16"
    ]
    print(f"Running benchmark on {device.upper()}...")
    try:
        subprocess.run(cmd, check=True, env={"PYTHONPATH": ".", **import_os_env()}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Failed to run on {device}.")
        return None
        
    duration = time.time() - start
    print(f"{device.upper()} completed {episodes} episodes in {duration:.2f} seconds ({duration/episodes:.2f} sec/ep).")
    return duration

def import_os_env():
    import os
    return os.environ.copy()

if __name__ == "__main__":
    print("--- PyTorch Device Benchmark ---")
    cpu_time = run_benchmark("cpu", episodes=2)
    mps_time = run_benchmark("mps", episodes=2)
    
    if cpu_time and mps_time:
        if cpu_time < mps_time:
            diff = (mps_time / cpu_time - 1) * 100
            print(f"\nVerdict: CPU is {diff:.1f}% FASTER than MPS for this Graph architecture due to memory transfer overhead!")
        else:
            diff = (cpu_time / mps_time - 1) * 100
            print(f"\nVerdict: MPS is {diff:.1f}% FASTER than CPU! Use hardware acceleration.")

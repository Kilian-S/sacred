#!/usr/bin/env python3
"""Launcher script to run SACRED coevolution training and monitor it via TensorBoard.

This script:
1. Launches the local TensorBoard server.
2. Automatically opens your default web browser to the TensorBoard dashboard.
3. Launches the ATLA training run.
4. Cleanly shuts down both processes when you press Ctrl+C.
"""

import os
import subprocess
import sys
import time
import webbrowser


def main() -> None:
    print("==================================================================")
    print("             SACRED RUN & MONITOR LAUNCHER                        ")
    print("==================================================================")

    # 1. Start TensorBoard
    print("\n[1/3] Spinning up TensorBoard server...")
    tb_cmd = [".venv/bin/tensorboard", "--logdir", "logs/tb_runs", "--port", "6006"]
    
    try:
        tb_process = subprocess.Popen(
            tb_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # Fallback if virtual env structure is slightly different
        tb_cmd[0] = "tensorboard"
        try:
            tb_process = subprocess.Popen(
                tb_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Error starting TensorBoard: {e}")
            sys.exit(1)

    # Give TensorBoard a moment to initialize
    time.sleep(2.5)

    # 2. Open default Web Browser
    print("[2/3] Opening TensorBoard dashboard in your web browser...")
    webbrowser.open("http://localhost:6006/")

    # 3. Start Training
    print("[3/3] Launching coevolutionary training loop...")
    print("------------------------------------------------------------------")
    
    # Pass along any extra arguments provided (e.g. --episodes 50)
    train_cmd = [".venv/bin/python", "scripts/train_sacred.py"] + sys.argv[1:]
    
    # Construct running environment, injecting PYTHONPATH
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    try:
        train_process = subprocess.Popen(train_cmd, env=env)
        
        # Keep launcher alive, monitoring training progress
        while True:
            ret_code = train_process.poll()
            if ret_code is not None:
                # Training finished or crashed
                if ret_code == 0:
                    print("\n[SUCCESS] Training run finished successfully!")
                else:
                    print(f"\n[ERROR] Training run crashed with exit code {ret_code}.")
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[INFO] Keyboard interrupt detected. Shutting down cleanly...")
    
    finally:
        # 4. Graceful Cleanup
        print("[CLEANUP] Stopping TensorBoard...")
        try:
            tb_process.terminate()
            tb_process.wait(timeout=3)
        except Exception:
            try:
                tb_process.kill()
            except Exception:
                pass

        try:
            train_process.terminate()
            train_process.wait(timeout=3)
        except Exception:
            try:
                train_process.kill()
            except Exception:
                pass

        print("Launcher stopped successfully. Have a great day!")


if __name__ == "__main__":
    main()

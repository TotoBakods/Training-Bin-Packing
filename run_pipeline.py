import subprocess
import os
import sys
import time

def run_command(cmd, log_file):
    print(f"Running: {cmd}")
    with open(log_file, "w", buffering=1) as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        process.wait()
    return process.returncode

def main():
    root = os.getcwd()
    
    # Step 1: Train
    train_rc = run_command("python gan/train.py", "train_run.log")
    if train_rc != 0:
        print(f"Training failed with RC {train_rc}")
        return

    # Step 2: Evaluate
    eval_rc = run_command("set PYTHONPATH=. && python gan/research_evaluator.py", "eval_run.log")
    if eval_rc != 0:
        print(f"Evaluation failed with RC {eval_rc}")
        return

    # Step 3: Generate
    gen_rc = run_command("python generate_training_data.py", "gen_run.log")
    if gen_rc != 0:
        print(f"Generation failed with RC {gen_rc}")
        return

    # Consolidate
    with open("full_pipeline_run.log", "w") as full:
        for log in ["train_run.log", "eval_run.log", "gen_run.log"]:
            if os.path.exists(log):
                full.write(f"\n--- {log} ---\n")
                with open(log, "r") as f:
                    full.write(f.read())
    
    print("Full pipeline complete!")

if __name__ == "__main__":
    main()

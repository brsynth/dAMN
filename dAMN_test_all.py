import subprocess
import os

# Modes and plot types to test
modes = ['forecast', 'medium']
plots = ['growth', 'substrate']

# Path to your test script
script_path = './dAMN_test.py'

# Loop over each mode and plot type
for mode in modes:
    for plot_type in plots:
        print(f"\n🔁 Running with mode='{mode}' and plot='{plot_type}'...")
        subprocess.run([
            "python3", script_path
        ], env={
            **os.environ,
            "train_test_split": mode,
            "plot": plot_type
        })

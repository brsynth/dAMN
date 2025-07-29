import subprocess
import os

# Modes and plot types to test
modes = ['forecast', 'medium']
plots = ['growth', 'substrate']
figure_folder = './figure_from_dAMN_test_all'
# Path to your test script
script_path = './dAMN_test.py'

os.makedirs(figure_folder, exist_ok=True)


# Loop over each mode and plot type
for mode in modes:
    for plot_type in plots:
        run_folder = f"{figure_folder}/{mode}_{plot_type}"
        os.makedirs(run_folder, exist_ok=True)

        print(f"\n Running with mode='{mode}' and plot='{plot_type}'...")
        subprocess.run([
            "python3", script_path
        ], env={
            **os.environ,
            "train_test_split": mode,
            "plot": plot_type,
            "figure_folder": run_folder
        })

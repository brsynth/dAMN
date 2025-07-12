# %% [markdown]
# ## Process data for NeuraldFBA

# %%
import pandas as pd
import numpy as np
from typing import List


def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def extract_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Extract column names that start with a given prefix."""
    return [col for col in df.columns if col.startswith(prefix)]


def get_unique_time_points(df: pd.DataFrame, time_columns: List[str]) -> List[float]:
    """Gather all unique time points from the dataset."""
    all_times = set()
    for col in time_columns:
        all_times.update(df[col].dropna().unique())
    return sorted(all_times)


def create_extended_dataframe(
    df: pd.DataFrame, time_cols: List[str], od_cols: List[str], dev_cols: List[str], all_times: List[float]
) -> pd.DataFrame:
    """Build a DataFrame with forward-filled missing OD and DEV values."""
    extended_frames = {}

    for t_col, od_col, dev_col in zip(time_cols, od_cols, dev_cols):
        mini_df = df[[t_col, od_col, dev_col]].dropna().set_index(t_col)
        mini_df.index.name = "Time"
        mini_df.columns = ["OD", "DEV"]

        # Reindex to full set of times and forward-fill missing values
        mini_df = mini_df.reindex(all_times).ffill()

        # Restore original column names
        mini_df.columns = [od_col, dev_col]
        extended_frames[(t_col, od_col, dev_col)] = mini_df

    return pd.concat(extended_frames.values(), axis=1)


def downsample_dataframe(df: pd.DataFrame, n_points: int, gamma: float = 2.0) -> pd.DataFrame:
    """
    Downsample the extended DataFrame to exactly `n_points` using a non-uniform
    sampling method that favors more points in the lag (early) phase.
    """
    min_time, max_time = df.index[0], df.index[-1]
    u = np.linspace(0, 1, n_points)
    u_trans = u ** gamma
    reduced_times = min_time + (max_time - min_time) * u_trans
    return df.reindex(reduced_times, method="nearest")


def reconstruct_original_format(df: pd.DataFrame, time_cols: List[str], od_cols: List[str], dev_cols: List[str]) -> pd.DataFrame:
    """Reconstruct the final DataFrame with T_i, OD_i, and DEV_i columns in alternating order."""
    df_out = df.reset_index()
    data = {}

    for i, (t_col, od_col, dev_col) in enumerate(zip(time_cols, od_cols, dev_cols), start=1):
        data[f"T_{i}"] = df_out["Time"]    # Copy time values into T_i columns
        data[f"OD_{i}"] = df_out[od_col]   # Copy OD values into OD_i columns
        data[f"DEV_{i}"] = df_out[dev_col] # Copy standard deviation values into DEV_i columns

    return pd.DataFrame(data)


def main(input_file: str = "OD.csv", output_file: str = "OD_clean.csv", n_points: int = 20, gamma: float = 2.0):
    """Main function to execute the workflow."""
    # Load data
    df_raw = load_data(input_file)

    # Extract relevant columns
    time_cols = extract_columns(df_raw, "T_")
    od_cols = extract_columns(df_raw, "OD_")
    dev_cols = extract_columns(df_raw, "DEV_")

    # Gather all unique time points
    all_times = get_unique_time_points(df_raw, time_cols)

    # Build full DataFrame with forward-filling
    df_extended = create_extended_dataframe(df_raw, time_cols, od_cols, dev_cols, all_times)

    # Downsample to exactly `n_points` using non-uniform sampling
    df_downsampled = downsample_dataframe(df_extended, n_points, gamma=gamma)

    # Reconstruct final output format
    df_final = reconstruct_original_format(df_downsampled, time_cols, od_cols, dev_cols)

    # Save to file
    df_final.to_csv(output_file, index=False)
    print(f"Processed data saved to '{output_file}'")


# User parameters
n_points = 10
gamma = 1.0  # Adjust gamma (>1) to cluster more points in the early (lag) phase.

main("./data/Paul_OD.csv", f"./data/Paul_OD_{str(n_points)}_{str(gamma)}.csv", n_points=n_points, gamma=gamma)


# %%
# Plot the formated growth curve

import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def load_extended_data(file_path: str) -> pd.DataFrame:
    """Load the processed OD data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' is not a valid CSV format.")
        exit(1)


def extract_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Extract column names that start with a given prefix."""
    return [col for col in df.columns if col.startswith(prefix)]


def plot_growth_curves(df: pd.DataFrame):
    """Plot OD (Optical Density) over time for each experiment."""
    time_cols = extract_columns(df, "T_")
    od_cols = extract_columns(df, "OD_")

    plt.figure(figsize=(10, 6))

    for t_col, od_col in zip(time_cols, od_cols):
        plt.plot(df[t_col], df[od_col], label=f"{t_col} / {od_col}", linestyle="-", marker="o", markersize=2)

    plt.xlabel("Time (min)")
    plt.ylabel("OD (log scale)")
    plt.title("E. coli Growth Curves")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside plot
    plt.grid(True, linestyle="--", alpha=0.6)  # Add a light grid for readability

    # Fix warning: Adjust right margin to fit legend
    plt.subplots_adjust(right=0.75)  

    plt.show()

"""Main function to execute the plotting workflow."""
file_path = f"./data/Paul_OD_10_1.0.csv"
df_plot = load_extended_data(file_path)
plot_growth_curves(df_plot)



# %%




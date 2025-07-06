import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# uniform, 20 points between 0 and 5 hours
common_times = np.linspace(0.0, 5.0, 20)

# Files to process
files = {
    "1mM" : "Millard_data_1mM_formated.csv",
    "10mM": "Millard_data_10mM_formated.csv",
    "30mM": "Millard_data_30mM_formated.csv"
}

# Interpolated results
interpolated_data = {}

for label, file in files.items():
    df = pd.read_csv(file)

    # Set up interpolation functions for each column (excluding 'T')
    interp_funcs = {
        col: interp1d(df['T'], df[col], kind='linear', fill_value='extrapolate')
        for col in df.columns if col != 'T'
    }

    # Build interpolated DataFrame
    interp_df = pd.DataFrame({'T': common_times})
    for col, f in interp_funcs.items():
        interp_df[col] = f(common_times)

    interpolated_data[label] = interp_df
    # Save the result
    interp_df.to_csv(f"Millard_data_{label}_interpolated.csv", index=False)

# Load Paul media to extract full set of metabolites
paul_media       = pd.read_csv("Paul_media.csv")
paul_metabolites = list(paul_media.columns[1:])  # exclude 'ID'

# Interpolated files
files = {
    1: "Millard_data_1mM_interpolated.csv",
    2: "Millard_data_10mM_interpolated.csv",
    3: "Millard_data_30mM_interpolated.csv"
}

media_rows = []
od_df_combined = pd.DataFrame()

for exp_id, file in files.items():
    df = pd.read_csv(file)
    initial = df.iloc[0]

    # Create media dictionary (fill missing metabolites with 0)
    media_dict = {
        'ID': exp_id,
        'glc__D_e': initial['GLC'],
        'acet_e': initial['ACE_env'],
        'accoa_c': initial['ACCOA'],
        'acp_c': initial['ACP'],
        'ac_c': initial['ACE_cell'],
    }
    for met in paul_metabolites:
        if met not in media_dict:
            media_dict[met] = 0.0
    media_rows.append(media_dict)

    # OD block: log-transform biomass (X)
    t_col = f'T_{exp_id}'
    od_col = f'OD_{exp_id}'
    dev_col = f'DEV_{exp_id}'

    tmp_df = pd.DataFrame({
        t_col: df['T'],
        od_col: np.log(df['X'] + 1e-8),  # log(OD)
        dev_col: np.zeros_like(df['X'])  # no deviation known
    })

    tmp_df.reset_index(drop=True, inplace=True)
    od_df_combined = pd.concat([od_df_combined, tmp_df], axis=1)

# Finalize DataFrames
media_df = pd.DataFrame(media_rows)
media_df = media_df[['ID'] + paul_metabolites]


# save
media_df.to_csv("Millard_media_from_interpolated.csv", index=False)
od_df_combined.to_csv("Millard_OD_from_interpolated.csv", index=False)


# Load the converted OD file
od_df = pd.read_csv("Millard_OD_from_interpolated.csv")

# List of experiments
exp_ids = [1, 2, 3]
labels = {
    1: "1 mM Acetate",
    2: "10 mM Acetate",
    3: "30 mM Acetate"
}

# Plot
plt.figure(figsize=(8, 5), dpi=300)
for exp_id in exp_ids:
    t_col = f'T_{exp_id}'
    od_col = f'OD_{exp_id}'

    if t_col in od_df.columns and od_col in od_df.columns:
        data = od_df[[t_col, od_col]].dropna()
        plt.plot(data[t_col], data[od_col], marker='o', label=labels.get(exp_id, f"ID {exp_id}"))

plt.title("log(OD) Timecourses for Millard Conditions")
plt.xlabel("Time (h)")
plt.ylabel("log(OD)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
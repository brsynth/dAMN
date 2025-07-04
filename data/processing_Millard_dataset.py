import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the formatted Millard datasets
df_1 = pd.read_csv("Millard_data_1mM_formated.csv")
df_10 = pd.read_csv("Millard_data_10mM_formated.csv")
df_30 = pd.read_csv("Millard_data_30mM_formated.csv")

# Load Paul media to get the full set of metabolites
paul_media = pd.read_csv("Paul_media.csv")
paul_metabolites = list(paul_media.columns[1:])  # exclude 'ID'

# Assign IDs for Millard datasets
id_map = {1: 1, 10: 2, 30: 3}
dfs = {1: df_1, 2: df_10, 3: df_30}

media_rows = []
od_df_combined = pd.DataFrame()

for label, df in dfs.items():
    initial = df.iloc[0]
    
    # Build the media row from initial concentrations
    media_dict = {
        'ID': label,
        'glc__D_e': initial['GLC'],
        'acet_e': initial['ACE_env'],
        'accoa_c': initial['ACCOA'],
        'acp_c': initial['ACP'],
        'ac_c': initial['ACE_cell'],
    }

    # Fill in other required metabolites with 0
    for met in paul_metabolites:
        if met not in media_dict:
            media_dict[met] = 0.0
    media_rows.append(media_dict)

    # Create OD block for each condition
    t_col = f'T_{label}'
    od_col = f'OD_{label}'
    dev_col = f'DEV_{label}'

    tmp_df = pd.DataFrame({
        t_col: df['T'],
        od_col: np.log(df['X'] + 1e-8),  # log-transform def: 1e-8
        dev_col: np.zeros_like(df['X'])  # no deviation data
    })

    # force to match expected row count (padding to 20 rows)
    while len(tmp_df) < 20:
        tmp_df.loc[len(tmp_df)] = [np.nan, np.nan, np.nan]

    tmp_df.reset_index(drop=True, inplace=True)
    od_df_combined = pd.concat([od_df_combined, tmp_df], axis=1)

# Finalize the media DataFrame and order columns
media_df = pd.DataFrame(media_rows)
media_df = media_df[['ID'] + paul_metabolites]

# Save results
media_df.to_csv("Millard_media_converted.csv", index=False)
od_df_combined.to_csv("Millard_OD_converted.csv", index=False)

# Load the OD file
od_df = pd.read_csv("Millard_OD_converted.csv")

# List experiment IDs to plot (e.g., 991, 992, 993 for Millard)
experiment_ids = [1, 2,3]

# Plot
plt.figure(figsize=(10, 6), dpi=300)
for exp_id in experiment_ids:
    time_col = f'T_{exp_id}'
    od_col = f'OD_{exp_id}'
    
    if time_col in od_df.columns and od_col in od_df.columns:
        od_df_clean = od_df[[time_col, od_col]].dropna()
        plt.plot(od_df_clean[time_col], od_df_clean[od_col],
                 label=f'ID {exp_id}', marker='o')

plt.xlabel('Time (h)')
plt.ylabel('log(OD)')
plt.title('Millard Simulated log(OD) Timecourses')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

    
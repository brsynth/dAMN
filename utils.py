import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf
import pandas as pd
import cobra
import json
import matplotlib.pyplot as plt
import sklearn


###############################################################################
# GENERAL UTILITIES
###############################################################################

OD_TO_CONC = 0.37
def concentration_to_OD(x):
    return np.log(np.maximum(np.array(x) / OD_TO_CONC, 1e-8))
    
def OD_to_concentration(x):
    return OD_TO_CONC * x  # resulting in gDW/L
    
def ReLU(x):
    return x * (x > 0)
    
def r2_growth_curve(
    Pred, Ref,
    OD=True,
):
    """
    Compute R² between predicted and reference growth curves for all experiments.
    Pred, Ref: (N_iter, N_exp, n_times, n_met)
    Returns: np.array of R² values (length N_exp)
    """
    from sklearn.metrics import r2_score
   
    N_iter, N_exp, n_times, n_met = Pred.shape
    assert Ref.shape == Pred.shape

    Pred_bio = Pred[..., -1]
    Ref_bio = Ref[..., -1]

    if OD:
        Pred_bio, Ref_bio = concentration_to_OD(Pred_bio), concentration_to_OD(Ref_bio)

    pred_bio_mean = np.mean(Pred_bio, axis=0)
    ref_bio_mean  = np.mean(Ref_bio, axis=0)

    R2 = []
    for i in range(N_exp):
        r2 = max(0, r2_score(ref_bio_mean[i], pred_bio_mean[i]))
        R2.append(r2)
    return np.array(R2)


###############################################################################
# PLOTTING UTILS
###############################################################################

def plot_loss(title, loss, num_epochs, save=''):
    """
    Generate a high-quality scientific plot for a single loss curve over training epochs.
    Parameters:
        title (str): Title of the plot.
        loss (array-like): Loss values (e.g., total loss or any individual loss).
        num_epochs (int): Number of epochs.
        save (str): Folder where the plot should be saved (as a PNG). If empty, plot is not saved.
    """
    
    # Set up scientific figure style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log(Loss)', color='blue')
    ax.plot(range(1, len(loss) + 1), loss, label='Loss', color='blue')
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelcolor='blue')

    plt.title(f'Loss over {num_epochs} Epochs ({title})', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    plt.tight_layout()
    # Save high-resolution figure if save path is provided
    if save != '':
        safe_title = title.replace(' ', '_')
        plt.savefig(f'{save}/loss_plot_{safe_title}.png', dpi=300, bbox_inches='tight')

def plot_predicted_reference_growth_curve(
    times, Pred, Ref,
    val_dev=None,
    OD=True, R2=None,
    train_time_steps=0,
    experiment_ids=None,
    run_name="run",
    train_test_split="medium",
    save=None
):
    """
    Plot all growth curves (OD or concentration) for multiple experiments/iterations.
    If train_time_steps is provided, splits the predicted curve at that point
    and draws a vertical line.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N_iter, N_exp, n_times, n_met = Pred.shape

    Pred_bio = Pred[..., -1]
    Ref_bio = Ref[..., -1]

    if OD:
        Pred_bio, Ref_bio = concentration_to_OD(Pred_bio), concentration_to_OD(Ref_bio)
        ylabel = "log(OD)"
    else:
        ylabel = "Concentration (mM)"

    pred_bio_mean, ref_bio_mean = np.mean(Pred_bio, axis=0), np.mean(Ref_bio, axis=0)
    pred_bio_std, ref_bio_std   = np.std(Pred_bio, axis=0), (val_dev if val_dev is not None else np.std(Ref_bio, axis=0))

    for i in range(N_exp):
        exp_label = f"Experiment {experiment_ids[i]}" if experiment_ids is not None else f"Exp {i}"
        title = f"{exp_label} {train_test_split}"

        plt.figure(figsize=(8, 6), dpi=300)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

        # Plot reference
        plt.plot(times, ref_bio_mean[i], marker='o', linestyle='-', color='black', label='Reference')

        # Plot prediction: split at train_time_steps if needed
        if 0 < train_time_steps < len(times):
            # Prediction up to forecast boundary
            plt.plot(times[:train_time_steps+1], pred_bio_mean[i][:train_time_steps+1],
                     marker='s', linestyle='--', color='darkgreen', label='Prediction')
            # Forecast region
            plt.plot(times[train_time_steps:], pred_bio_mean[i][train_time_steps:],
                     marker='s', linestyle='--', color='darkgreen')
            # Connector
            plt.plot(
                [times[train_time_steps-1], times[train_time_steps]],
                [pred_bio_mean[i][train_time_steps-1], pred_bio_mean[i][train_time_steps]],
                linestyle='-', color='darkgreen', linewidth=3, alpha=0.6
            )
            plt.axvline(times[train_time_steps], color='k', 
                        linestyle=':', alpha=0.6, label='Forecast Start')
        else:
            plt.plot(times, pred_bio_mean[i], marker='s', 
                     linestyle='--', color='darkgreen', label='Prediction')

        if ref_bio_std is not None:
            plt.fill_between(times, ref_bio_mean[i] - ref_bio_std[i], 
                             ref_bio_mean[i] + ref_bio_std[i], color='black', alpha=0.2)
        if pred_bio_std is not None:
            plt.fill_between(times, pred_bio_mean[i] - pred_bio_std[i], 
                             pred_bio_mean[i] + pred_bio_std[i], color='darkgreen', alpha=0.2)

        plt.xlabel('Time (h)')
        plt.ylabel(ylabel)
        plt.title(title, pad=20)
        
        # ---- Print R2 on plot ----
        if R2 is not None:
            plt.text(0.05, 0.95, f"R²: {R2[i]:.2f}", transform=plt.gca().transAxes,
                     fontsize=14, verticalalignment='top', 
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        plt.grid(True, linestyle='--', alpha=0.6)
        #plt.legend(frameon=True, loc='best', fancybox=True)
        plt.tight_layout()

        if save:
            title_clean = f"{title.replace(' ', '_').replace('/', '')}"
            plt.savefig(f'{save}/{title_clean}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_predicted_biomass_and_substrate(
    times, Pred,
    experiment_ids=None,
    metabolite_ids=None,
    run_name="run",
    train_test_split="medium",
    save=None
):
    """
    For each experiment, plot (on the same figure) the predicted growth curve (biomass, gDW/L) and
    the predicted substrate concentration (mM, substrate index is detected per experiment).
    Only Predicted values are plotted.
    Pred: (N_iter, N_exp, n_times, n_met)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N_iter, N_exp, n_times, n_met = Pred.shape

    # --- Find substrate indices ---
    def substrate_id(data):
        valid_mask = (~np.isnan(data)) & (data != 0)
        first_valid_indices = np.full((data.shape[0], data.shape[1]), -1)
        for i in range(data.shape[0]):
            for t in range(data.shape[1]):
                valid = valid_mask[i, t]
                if np.any(valid):
                    first_valid_indices[i, t] = np.argmax(valid)
        id = []
        for i in range(data.shape[0]):
            found = first_valid_indices[i][first_valid_indices[i] != -1]
            id.append(found[0] if len(found) else -1)
        return id

    sub_ids = substrate_id(Pred[0])   # shape (N_exp,)

    # --- Extract BIOMASS and SUBSTRATE curves ---
    Pred_bio = Pred[..., -1]
    Pred_sub = np.zeros((N_iter, N_exp, n_times))
    for i in range(N_exp):
        idx = sub_ids[i]
        if idx < 0:
            Pred_sub[:, i, :] = np.nan
        else:
            Pred_sub[:, i, :] = Pred[:, i, :, idx]

    # --- Mean and std across N_iter (axis=0) ---
    pred_bio_mean = np.mean(Pred_bio, axis=0)
    pred_bio_std = np.std(Pred_bio, axis=0)
    pred_sub_mean = np.mean(Pred_sub, axis=0)
    pred_sub_std = np.std(Pred_sub, axis=0)

    # --- Plot ---
    for i in range(N_exp):
        exp_label = f"Experiment {experiment_ids[i]}" if experiment_ids is not None else f"Exp {i}"
        sub_name = metabolite_ids[sub_ids[i]] if (metabolite_ids is not None and sub_ids[i] >= 0) else "Unknown"
        title = f"{exp_label} {train_test_split} {sub_name}"
        print(f"Experiment {exp_label}: Substrate = {sub_name} (index {sub_ids[i]})")

        # 1. Use subplots and twinx for dual y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

        # 2. Plot biomass (left axis)
        ax1.plot(times, pred_bio_mean[i], marker='s', linestyle='--', color='darkgreen', label='Pred Biomass')
        ax1.fill_between(times, pred_bio_mean[i] - pred_bio_std[i], pred_bio_mean[i] + pred_bio_std[i], color='darkgreen', alpha=0.2)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel("Biomass (gDW/L)", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # 3. Plot substrate (right axis)
        ax2 = ax1.twinx()
        ax2.plot(times, pred_sub_mean[i], marker='v', linestyle='--', color='orange', label=f'Pred Substrate ({sub_name})')
        ax2.fill_between(times, pred_sub_mean[i] - pred_sub_std[i], pred_sub_mean[i] + pred_sub_std[i], color='orange', alpha=0.15)
        ax2.set_ylabel(f"{sub_name} (mM)", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # 4. Title and grid (on ax1)
        # plt.title(title, pad=20)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 5. Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines1 + lines2, labels1 + labels2, frameon=True, loc='best', fancybox=True)

        plt.tight_layout()

        if save:
            title_clean = f"{title.replace(' ', '_').replace('/', '')}"
            plt.savefig(f'{save}/{title_clean}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_similarity_distribution(title, r2_values, save=''):
    """
    Plot R2 value distribution
    """
    r2_values = np.asarray(r2_values)
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2
    })

    # ---- Set bins with fixed width 0.05
    bin_width = 0.1
    min_val = np.floor(r2_values.min() / bin_width) * bin_width
    max_val = np.ceil(r2_values.max() / bin_width) * bin_width
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    n, bins, patches = plt.hist(r2_values, bins=bins, color='grey', edgecolor='black', alpha=0.7, density=True)
    
    # Convert density to percent
    bin_widths = np.diff(bins)
    n_percent = n * bin_widths * 100  # now n_percent sums to 100%
    plt.clf()  # Clear to redraw

    # Plot again with percent
    plt.bar(bins[:-1], n_percent, width=bin_widths, color='grey', edgecolor='black', alpha=0.7, align='edge')
    plt.xlabel('R2')
    plt.ylabel('Frequency (%)')
    #plt.title(f"{title}\nR2={mean_r2:.2f}±{std_r2:.2f}", pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save != '':
        safe_title = title.replace(' ', '_')
        plt.savefig(f'{save}/{safe_title}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
###############################################################################
# PROCESS DATA: MEDIA, OD, AND COBRA MODEL
###############################################################################

def build_Stoichiometry_and_Transport_from_cobra(
    cobra_model_file: str, 
    medium_met_ids: List[str], 
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[np.ndarray, np.ndarray, int, int, List[str]]:
    """
    Build stoichiometric and Transport matrices from a COBRA model.
    """
    model = cobra.io.read_sbml_model(cobra_model_file)
    
    # Get reaction and metabolite IDs (strings)
    rxn_ids = [rxn.id for rxn in model.reactions]
    full_met_ids = [met.id for met in model.metabolites]

    m = len(full_met_ids)
    n = len(rxn_ids)

    # Build stoichiometric matrix
    Stoichiometry = np.zeros((m, n), dtype=np.float32)
    for j, rxn in enumerate(model.reactions):
        for met, coeff in rxn.metabolites.items():
            i = full_met_ids.index(met.id)
            Stoichiometry[i, j] = coeff
            
    # Build Transport matrix
    Transport = np.zeros((len(medium_met_ids), n), dtype=np.float32)
    for col_idx, rxn_obj in enumerate(model.reactions):
        rxn_id = rxn_obj.id
        if rxn_id == biomass_rxn_id:
            row_idx = medium_met_ids.index('BIOMASS')
            Transport[row_idx, col_idx] = 1.0
            if verbose:
                print(f'Transport[{row_idx},{col_idx}] = 1.0  BIOMASS')
        elif rxn_id.startswith('EX_'):
            io = rxn_id[-1:]
            met_id_candidate = rxn_id[3:-2]  # e.g. 'EX_glc__D_e' -> 'glc__D'
            if met_id_candidate in medium_met_ids:
                row_idx = medium_met_ids.index(met_id_candidate)
                Transport[row_idx, col_idx] = -1.0 if io == 'i' else 1.0 
                if verbose:
                    print(f'Transport[{row_idx},{col_idx}] = {Transport[row_idx, col_idx]}  {rxn_id}')

    if verbose:
        print(f'Transport shape: {Transport.shape}, Stoichiometry shape: {Stoichiometry.shape}')
        print(f'Number of metabolites (k): {len(medium_met_ids)}, Number of fluxes (n): {n}')
    return Stoichiometry, Transport, m, n, rxn_ids

def process_data(
    media_file: str,
    od_file: str,
    cobra_model_file: str,
    biomass_rxn_id: str,
    verbose=False
) -> Tuple[np.ndarray, List[str], Dict[int, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Process media, OD, and COBRA model to generate structured experimental data and matrices.
    """
    media_df = pd.read_csv(media_file)
    od_df = pd.read_csv(od_file)

    # Extract time points from columns like 'T_...'
    times = sorted(pd.unique(od_df.filter(like='T').values.flatten()))
    times = [t for t in times if not pd.isnull(t)]

    # Use all media metabolites (skip the first column 'ID') and add BIOMASS at the end.
    metabolite_ids = list(media_df.columns[1:])
    metabolite_ids.append('BIOMASS')
    
    experiment_data, dev_data = {}, {}
    for exp_id in media_df['ID']:
        t_col = f'T_{int(exp_id)}'
        od_col = f'OD_{int(exp_id)}'
        dev_col = f'DEV_{int(exp_id)}'
    
        if t_col not in od_df.columns or od_col not in od_df.columns:
            raise ValueError(f'Missing columns for experiment ID {exp_id} in OD file.')

        od_times = od_df[t_col].values
        log_od_values = od_df[od_col].values
        log_od_dev = od_df[dev_col].values 
        dev_data[int(exp_id)] = log_od_dev

        od_values = np.exp(log_od_values)
        biomass_concentration = OD_to_concentration(od_values)

        conc_matrix = np.full((len(times), len(metabolite_ids)), np.nan, dtype=np.float32)
        # Set initial media concentrations (excluding biomass)
        row_in_media = media_df[media_df['ID'] == exp_id].iloc[0, 1:].values
        conc_matrix[0, :-1] = row_in_media

        # Set initial biomass from OD data if available
        if 0 in od_times:
            idx_0 = np.where(od_times == 0)[0][0]
            conc_matrix[0, -1] = biomass_concentration[idx_0]
        else:
            conc_matrix[0, -1] = biomass_concentration[0]

        # Fill in biomass for subsequent time points
        for i_time, t_val in enumerate(times[1:], start=1):
            if t_val in od_times:
                idx_t = np.where(od_times == t_val)[0][0]
                conc_matrix[i_time, -1] = biomass_concentration[idx_t]

        experiment_data[int(exp_id)] = conc_matrix
 
    # Build stoichiometric and transport matrices
    Stoichiometry, Transport, _, _, rxn_ids = build_Stoichiometry_and_Transport_from_cobra(
        cobra_model_file, metabolite_ids, biomass_rxn_id, verbose=verbose
    )
    return np.array(times), metabolite_ids, experiment_data, dev_data, Stoichiometry, Transport, rxn_ids

def prepare_experiment_array(
    time_ids: np.ndarray,
    metabolite_ids: List[str],
    experiment_data: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Flatten experimental data into a 2D array with shape: (num_experiments, (T+1)*k).
    """
    k = len(metabolite_ids)
    all_flat = []
    for exp_id in experiment_data.keys():
        conc_matrix = experiment_data[exp_id]
        row_flat = conc_matrix[:time_ids].reshape(-1)
        all_flat.append(row_flat)
    return np.stack(all_flat, axis=0)

###############################################################################
# MODEL with BATCHED TIME-UNROLL and ACTUAL TIME UPDATES
###############################################################################

class LagNetwork(tf.keras.layers.Layer):
    """
    Neural network for learning lag-phase parameters from initial conditions.
    This layer predicts two parameters for each trajectory:
      - t_lag:   lag time (duration of lag phase)
      - r_lag:   stiffness parameter for the sigmoid ramp
    """
    def __init__(self, input_dim, hidden_layers=[50], dropout_rate=0.2, name='lag_network', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(h, activation='relu', name=f'lag_hidden_{i}')
            for i, h in enumerate(hidden_layers)
        ])
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='lag_dropout')
        self.out_layer = tf.keras.layers.Dense(2, activation=None, name='lag_output')  # 2 outputs: t_lag and r_lag

    def call(self, x, training=False):
        x = self.hidden_layers(x)
        x = self.dropout(x, training=training)
        out = self.out_layer(x)
        # For stability: softplus for r_lag (so it's positive), softplus or relu for t_lag (positive)
        t_lag = tf.nn.softplus(out[..., 0:1])  # shape (batch, 1)
        r_lag = tf.nn.softplus(out[..., 1:2])  # shape (batch, 1)
        return t_lag, r_lag

class FluxNetwork(tf.keras.layers.Layer):
    """
    Neural network for learning fluxes from concentration
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[500], dropout_rate=0.2, name='flux_network', **kwargs):
        super().__init__(name=name, **kwargs)
        # Use Sequential to ensure Keras tracks all sublayers
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                units=h, 
                activation='relu', 
                name=f'hidden_dense_{i}'
            ) for i, h in enumerate(hidden_layers)
        ], name='hidden_layers')

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        self.out_layer = tf.keras.layers.Dense(
            units=output_dim, 
            activation='linear', 
            name='output_dense'
        )

    def call(self, x, training=False):
        x = self.hidden_layers(x)
        x = self.dropout(x, training=training)
        return self.out_layer(x)

class MetabolicModel(tf.keras.Model):
    """
    Model that unrolls the time dimension.
    Expects input shape: (batch_size, T+1, k)
    dt_vector: tensor of shape (T,) with time differences (t[i+1]-t[i])
    """

    def __init__(self, times, metabolite_ids, Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
        hidden_layers_lag=[0], hidden_layers_flux=[500], dropout_rate=0.2, 
        loss_weight=[0, 0, 0, 0], loss_decay=[0, 0, 0, 0],
        train_test_split='medium', x_fold=5,
        train_time_steps=0,
        verbose=True, name='metabolic_model'):

        super().__init__(name=name)
        self.times = times
        self.metabolite_ids = metabolite_ids
        self.T = len(times)  # T represents the number of intervals
        self.train_time_steps = train_time_steps if train_time_steps > 0 else T
        dt_vector = np.diff(times).astype(np.float32)
        dt_vector = dt_vector / np.min(dt_vector)
        self.dt_vector = tf.convert_to_tensor(dt_vector, dtype=tf.float32)
        self.Transport = Transport
        self.Stoichiometry = Stoichiometry
        self.k, self.n = Transport.shape[0], Transport.shape[1]
        self.rxn_ids = rxn_ids
        self.biomass_rxn_id = biomass_rxn_id
        self.biomass_flux_index = rxn_ids.index(biomass_rxn_id)
        self.hidden_layers_lag = hidden_layers_lag
        self.hidden_layers_flux = hidden_layers_flux
        self.dropout_rate = dropout_rate
        self.loss_weight = loss_weight
        self.loss_decay = loss_decay
        self.train_test_split = train_test_split
        self.x_fold = x_fold
        
        # Add lag network: maps initial condition to (t_lag, r_lag)
        if self.hidden_layers_lag[0] > 0:
            self.lag_net = LagNetwork(self.k, hidden_layers=hidden_layers_lag, dropout_rate=dropout_rate, name='lag_network')
        # Initialize FluxNetwork  
        self.flux_net = FluxNetwork(self.k, self.n, hidden_layers_flux, dropout_rate, name='flux_network')

    def build(self, input_shape):
        super().build(input_shape)
        # No additional build steps are necessary since all layers are initialized in __init__

    def printout(self):
        print(f'-----------------------------MetabolicModel-----------------------------')
        print(f'times: {self.times[0]:.2f}, {self.times[1]:.2f}, ..., {self.times[-1]:.2f}')
        print(f'metabolite_ids: {self.metabolite_ids}')
        print(f'Total time step: {self.T}')
        print(f'Train time step: {self.train_time_steps}')
        print(f'dt: {self.dt_vector.numpy()}')
        print(f'Transport: {self.Transport.shape}')
        print(f'Stoichiometry: {self.Stoichiometry.shape}')
        print(f'n: {self.n}')
        print(f'k: {self.k}')
        print(f'Reaction ids: {len(self.rxn_ids)}')
        print(f'Biomass id: {self.biomass_rxn_id}')
        print(f'Biomass flux index: {self.biomass_flux_index}')
        if  self.hidden_layers_lag[0] > 0:
            print(f'Lag Layer : Hidden size = {self.hidden_layers_lag} trainainle parameters = {self.lag_net.count_params()}')
        print(f'Flux Layer : Hidden size = {self.hidden_layers_flux} trainainle parameters = {self.flux_net.count_params()}')
        print(f'Dropout Rate: {self.dropout_rate}')
        print(f'Loss weight: {self.loss_weight}')
        print(f'Loss decay: {self.loss_decay}')
        print(f'train_test_split: {getattr(self, "train_test_split", "N/A")}')
        print(f'train_test_split: {getattr(self, "train_time_steps", "N/A")}')
        print(f'x_fold: {getattr(self, "x_fold", "N/A")}')
        print(f'------------------------------------------------------------------------')

    def debug_concentration_step(self, t, C_pred, C_next, delta_C, v_t, lag_params, rt, verbose):
        """Prints debug information for concentration updates."""
        """we print experiment I == verbose-1"""

        if len(self.metabolite_ids) == 0:
            return
        I = verbose-1 # expeimrnat to be printed
        tf.print(f't={t}------------------------------------------------------------------')
        t_lag = lag_params['t_lag']
        r_lag = lag_params['r_lag']
        t_lag = tf.reshape(t_lag, [-1, 1])
        r_lag = tf.reshape(r_lag, [-1, 1])    
        print(f"Experiment {I+1}: t_lag = {t_lag[I, 0].numpy():.4f}, r_lag = {r_lag[I, 0].numpy():.4f} r_t={float(rt[I])}")

        # Print non-zero C_pred (for t=0)
        if float(t) == 0.0 and self.metabolite_ids is not None:
            nonzero_mask_pred = tf.not_equal(C_pred[I], 0.0)
            nonzero_indices_pred = tf.where(nonzero_mask_pred)
            nonzero_values_pred = tf.gather(C_pred[I], nonzero_indices_pred)
            tf.print("C_pred non-zero (t=0):")
            for idx, val in zip(tf.reshape(nonzero_indices_pred, [-1]).numpy(), tf.reshape(nonzero_values_pred, [-1]).numpy()):
                met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
                print(f'  idx={idx}  {met_name} = {val:.2f}')

        # print v_t statistics
        tf.print(f'v_t stats min: {tf.reduce_min(v_t[I]):.2f} ' \
                 f'max: {tf.reduce_max(v_t[I]):.2f} ' \
                 f'mean : {tf.reduce_mean(v_t[I]):.2f} ' \
                 f'num NaN: {tf.reduce_sum(tf.cast(tf.math.is_nan(v_t[I]), tf.int32))}')

        # Print non-zero delta_C C_next
        nonzero_mask_next = tf.not_equal(C_next[I], 0.0)
        nonzero_indices_next = tf.where(nonzero_mask_next)
        nonzero_values_next = tf.gather(C_next[I], nonzero_indices_next)
        nonzero_values_delta = tf.gather(delta_C[I], nonzero_indices_next)
        tf.print("delta_C non-zero:")
        for idx, val in zip(tf.reshape(nonzero_indices_next, [-1]).numpy(), tf.reshape(nonzero_values_delta, [-1]).numpy()):
            met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
            print(f'  idx={idx}  {met_name} = {val:.2f}')
        tf.print("C_next non-zero:")
        for idx, val in zip(tf.reshape(nonzero_indices_next, [-1]).numpy(), tf.reshape(nonzero_values_next, [-1]).numpy()):
            met_name = self.metabolite_ids[idx] if idx < len(self.metabolite_ids) else f"Met{idx}"
            print(f'  idx={idx}  {met_name} = {val:.2f}')

    def _compute_lag_params(self, C_pred, t, training, lag_params=None, verbose=False):
        """
        Compute r_t_broadcast and lag_params using exponential approach.
        Returns: r_t_broadcast, lag_params
        """
        if self.hidden_layers_lag[0] > 0:
            if lag_params is None:
                t_lag, r_lag = self.lag_net(C_pred, training=training)
                lag_params = dict(t_lag=t_lag, r_lag=r_lag)
            else:
                t_lag = lag_params['t_lag']
                r_lag = lag_params['r_lag']

            # Ensure shape: (batch_size, 1)
            t_lag = tf.reshape(t_lag, [-1, 1])
            r_lag = tf.reshape(r_lag, [-1, 1])

            # t may be scalar or tensor; make compatible with batch
            if not tf.is_tensor(t):
                t = tf.convert_to_tensor(t, dtype=tf.float32)
            t = tf.reshape(t, [1, 1])  # (1, 1)
            t = tf.broadcast_to(t, tf.shape(t_lag))  # (batch_size, 1)

            # Exponential lag formula: r_t = (1 - exp(-r_lag * t)) / (1 - exp(-r_lag * t_lag))
            numerator = 1.0 - tf.exp(-r_lag * t)
            denominator = 1.0 - tf.exp(-r_lag * t_lag)
            # Prevent division by zero (when r_lag * t_lag is very small)
            denominator = tf.where(tf.abs(denominator) < 1e-8, tf.ones_like(denominator), denominator)
            r_t_exp = numerator / denominator
            r_t = tf.where(t < t_lag, r_t_exp, tf.ones_like(r_t_exp))
            r_t = tf.clip_by_value(r_t, 0.0, 1.0)
            r_t_broadcast = r_t  # shape (batch_size, 1)
        else:
            r_t_broadcast = 1.0
            lag_params = None
        
        return r_t_broadcast, lag_params

    def _next_concentration(self, C_pred, t, training, lag_params=None, verbose=False):
        """
        Compute the next concentration vector, including lag-phase, for any time step.
        """
        r_t, lag_params = self._compute_lag_params(C_pred, t, training, lag_params, verbose=verbose)
        v_t = self.flux_net(C_pred, training=training)
        delta_C = tf.matmul(v_t, self.Transport, transpose_b=True)
        t_idx = int(t) if isinstance(t, (int, np.integer)) else int(t.numpy())
        dt = self.dt_vector[-1] if t_idx >= self.dt_vector.shape[0] else self.dt_vector[t_idx]
        C_next = C_pred + r_t * delta_C * dt #  tf.nn.relu(C_pred + r_t_broadcast * delta_C * dt)

        # Call debug function
        if verbose:
            self.debug_concentration_step(t, C_pred, C_next, v_t, delta_C, lag_params, r_t, verbose)

        return C_next, v_t, lag_params

    def compute_losses(self, v_t, C_pred, C_next, C_ref_next, loss_weight, decay):
        """
        Compute the 4 loss components given the current state.
        """
        # Loss on SV
        loss = tf.matmul(v_t, self.Stoichiometry, transpose_b=True)  # shape: (batch_size, m)
        loss = tf.reduce_mean(tf.square(loss))
        loss_s_v = loss_weight[0] * decay[0] * loss

        # Loss to penalize negative fluxes
        loss = tf.reduce_mean(tf.nn.relu(-v_t))
        loss_neg_v = loss_weight[1] * decay[1] * loss

        # Loss on concentration
        mask = ~tf.math.is_nan(C_ref_next)
        loss = tf.reduce_mean(tf.square(tf.boolean_mask(C_next, mask) - tf.boolean_mask(C_ref_next, mask)))
        loss_c = loss_weight[2] * decay[2] * loss

        # Loss to penalize drop in biomass concentration
        loss = tf.reduce_mean(tf.nn.relu(C_pred[:,-1]-C_next[:,-1]))
        loss_drop_c = loss_weight[3] * decay[3] * loss

        return loss_s_v, loss_neg_v, loss_c, loss_drop_c

    
    def call(self, C_ref_batch, training=False, verbose=False):
        """
        C_ref_batch: shape (batch_size, T+1, k)
        dt_vector: shape (T,) containing dt for each time interval.
        """
        batch_size = tf.shape(C_ref_batch)[0]
        T = tf.shape(C_ref_batch)[1]
        t_max = tf.cast(T, tf.float32)
        C0 = C_ref_batch[:, 0, :]   # shape (batch_size, k)
        lag_params = None
        C_pred = C_ref_batch[:, 0, :]  # shape: (batch_size, k)
        loss_s_v, loss_neg_v, loss_c, loss_drop_c  = 0.0, 0.0, 0.0, 0.0
        verbose = False if training else verbose
        
        for t in tf.range(T-1):

            ttf = tf.cast(t, tf.float32)

            C_next, v_t, lag_params = self._next_concentration(
            C_pred, ttf, training=training, lag_params=lag_params, verbose=verbose
            )
 
            # Reference concentrations for time t+1, shape: (batch_size, k)
            C_ref_next = C_ref_batch[:, t+1, :]
            
            # Compute all 4 losses at this step
            decay = [tf.exp(-self.loss_decay[i] * ttf) for i in range(4)]
            ls_v, lnv, lc, ldc = self.compute_losses(
            v_t, C_pred, C_next, C_ref_next,  self.loss_weight, decay
            )
            loss_s_v += ls_v
            loss_neg_v += lnv
            loss_c += lc
            loss_drop_c += ldc

            C_pred = C_next

        return loss_s_v, loss_neg_v, loss_c, loss_drop_c
        
    def save_model(self, model_name='dAMNmodel', verbose=False):
        """
        Save the model weights and configuration (architecture parameters) to files.
        """
        weights_path = f'{model_name}.weights.h5'
        config_path = f'{model_name}.config.json'

        # Ensure model is built before saving by running a dummy forward pass
        dummy_input = tf.random.normal((1, len(self.times), self.k))  # Shape: (batch_size, T+1, k)
        _ = self(dummy_input)  # Forces all layers to initialize

        # Save the weights
        self.save_weights(weights_path)

        # Save model configuration
        config = {
        'times': self.times.tolist() if isinstance(self.times, (np.ndarray, list)) else list(self.times),
        'train_time_steps': int(self.train_time_steps),
        'metabolite_ids': list(self.metabolite_ids),
        'Transport': self.Transport.tolist(),
        'Stoichiometry': self.Stoichiometry.tolist(),
        'k': int(self.k),
        'n': int(self.n),
        'rxn_ids': self.rxn_ids,
        'biomass_rxn_id': self.biomass_rxn_id,
        'biomass_flux_index': int(self.biomass_flux_index),
        'hidden_layers_lag': self.hidden_layers_lag,
        'hidden_layers_flux': self.hidden_layers_flux,
        'dropout_rate': float(self.dropout_rate),
        'loss_weight': self.loss_weight,
        'loss_decay': self.loss_decay,
        'train_test_split': getattr(self, "train_test_split", "medium"),
        'x_fold': getattr(self, "x_fold", 5)
        }
    
        # Write config to JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f)

        if verbose:
            print(f'Model weights saved to {weights_path} and config saved to {config_path}')

    @classmethod
    def load_model(cls, model_name='dAMNmodel', verbose=False):
        """
        Load a model from saved weights and configuration.
        """
        weights_path = f'{model_name}.weights.h5'
        config_path = f'{model_name}.config.json'

        # Check if files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Configuration file {config_path} not found.')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights file {weights_path} not found.')

        # Load the configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract parameters
        times = config['times']
        T = len(times)
        train_time_steps = config.get('train_time_steps', T)
        metabolite_ids = list(config.get('metabolite_ids', []))
        Transport = np.array(config['Transport'])
        Stoichiometry = np.array(config['Stoichiometry'])
        rxn_ids = config['rxn_ids']
        biomass_rxn_id = config['biomass_rxn_id']
        hidden_layers_lag = config['hidden_layers_lag']
        hidden_layers_flux = config['hidden_layers_flux']
        dropout_rate = config['dropout_rate']
        loss_weight = config['loss_weight']
        loss_decay = config['loss_decay']
        train_test_split = config.get('train_test_split', 'medium')
        x_fold = config.get('x_fold', 5)
        
        # Create a new model instance
        model = cls(
        times=times,
        metabolite_ids=metabolite_ids,
        train_time_steps=train_time_steps,
        Transport=Transport,
        Stoichiometry=Stoichiometry,
        rxn_ids=rxn_ids,
        biomass_rxn_id=biomass_rxn_id,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        dropout_rate=dropout_rate,
        loss_weight = loss_weight,
        loss_decay = loss_decay,
        train_test_split=train_test_split,
        x_fold=x_fold,
        verbose=verbose,
        name='metabolic_model'
        )
        
        # Ensure the model is built before loading weights
        dummy_input = tf.random.normal((1, T, model.k))  # Shape: (batch_size, T, k)
        _ = model(dummy_input)  # Forces all layers to initialize

        # Load saved weights
        model.load_weights(weights_path)
        if verbose:
            print(f'Model built from {config_path} and weights loaded from {weights_path}')
            
        return model

###############################################################################
# CREATE TRAIN AND CROSS-VALIDATE MODEL
###############################################################################

def reset_model(
    model,
    verbose=False
):
    """
    Reset provided model: returns a new model instance with same architecture and config.
    """

    # Use model attributes directly
    new_model = MetabolicModel(
        times=model.times,
        metabolite_ids=model.metabolite_ids,
        Transport=model.Transport,
        Stoichiometry=model.Stoichiometry,
        rxn_ids=model.rxn_ids,
        biomass_rxn_id=model.biomass_rxn_id,
        hidden_layers_lag=model.hidden_layers_lag,
        hidden_layers_flux=model.hidden_layers_flux,
        dropout_rate=model.dropout_rate,
        loss_weight=model.loss_weight,
        loss_decay=model.loss_decay,
        train_test_split=model.train_test_split,
        x_fold=model.x_fold,
        train_time_steps=model.train_time_steps,
        verbose=verbose
    )

    # Force model to build
    dummy = np.zeros((1, len(model.times), len(model.metabolite_ids)), dtype=np.float32)
    _ = new_model(dummy, training=False)
    if verbose:
        new_model.printout()

    return new_model
    
def create_model_train_val(
    media_file, od_file,
    cobra_model_file,
    biomass_rxn_id,
    x_fold=5, 
    hidden_layers_lag=[0],
    hidden_layers_flux=[460], 
    dropout_rate=0.2,
    loss_weight=[0.001, 1, 1, 1],
    loss_decay=[0, 0.5, 0.5, 0.5],
    train_test_split='medium',
    verbose=False
    ):
    """
    Prepares training and validation data and model for two split strategies:
      - 'medium': Split by random subsets of media/conditions (current default).
      - 'forecast': Temporal split within each medium, training on the first fraction
                   of time steps, validating (forecasting) on the last.
    Parameters
    ----------
    train_test_split: 'medium' or 'forecast'
        'medium': traditional split by media (default)
        'forecast': train/test split along time axis within all media
    x_fold: int
        For 'medium', determines size of validation set (1/x_fold of media).
        For 'forecast', training uses (x_fold-1)/x_fold of time steps, testing uses 1/x_fold.
    Returns
    -------
    model, train_array, train_dev, val_array, val_dev, val_ids
        If 'forecast', val_ids will be time indices used for test set.
    """
    # Load and process data 
    times, metabolite_ids, experiment_data, dev_data, Stoichiometry, Transport, rxn_ids = process_data(
        media_file, od_file, cobra_model_file, biomass_rxn_id, verbose=verbose
    )
    exp_ids = list(experiment_data.keys())
    total_time_steps = len(times)
    
    if train_test_split == 'medium': # Splitting by media
        np.random.shuffle(exp_ids)
        split = int(len(exp_ids) * (1 - 1/x_fold)) if x_fold > 1 else len(exp_ids)
        train_time_steps = total_time_steps
        train_ids = exp_ids[:split]
        train_data = {eid: experiment_data[eid] for eid in train_ids}
        train_array = prepare_experiment_array(total_time_steps, metabolite_ids, train_data)
        dev_data_arr = np.asarray(list(dev_data.values()))
        train_dev = dev_data_arr[:split]
        if x_fold > 1:
            val_ids = exp_ids[split:]
            val_data = {eid: experiment_data[eid] for eid in val_ids}
            val_array = prepare_experiment_array(total_time_steps, metabolite_ids, val_data)
            val_dev = dev_data_arr[split:]
        else:
            val_array, val_dev, val_ids = train_array , train_dev, train_ids      

    elif train_test_split == 'forecast':
        # Splitting along time axis
        train_time_steps = int(round(total_time_steps * (x_fold - 1) / x_fold)) if x_fold > 1 else total_time_steps
        train_ids = exp_ids
        train_data = {eid: experiment_data[eid] for eid in train_ids}
        train_array = prepare_experiment_array(train_time_steps, metabolite_ids, train_data)
        dev_data_arr = np.asarray(list(dev_data.values()))
        train_dev = dev_data_arr[:,:train_time_steps]
        if x_fold > 1:
            val_ids = train_ids
            val_array = prepare_experiment_array(total_time_steps, metabolite_ids, train_data)
            val_dev = dev_data_arr           
        else:
            val_array, val_dev, val_ids = train_array , train_dev, train_ids          
    else:
        raise ValueError(f"Unknown train_test_split value: {train_test_split}")

    # Instantiate the model 
    model = MetabolicModel(
        times, 
        metabolite_ids,
        Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
        hidden_layers_lag=hidden_layers_lag, 
        hidden_layers_flux=hidden_layers_flux, 
        dropout_rate=dropout_rate, 
        loss_weight=loss_weight, 
        loss_decay=loss_decay,
        train_test_split=train_test_split, 
        x_fold=x_fold,
        train_time_steps=train_time_steps,
        verbose=verbose
    )

    if verbose:
        print(f'Train shape: {train_array.shape}')
        print(f'Val shape: {val_array.shape}')
        dummy = np.zeros((1, model.T, model.k), dtype=np.float32)
        _ = model(dummy, training=False)
        model.printout()
        
    return model, train_array, train_dev, val_array, val_dev, val_ids

def train_step(model, C_ref_batch, optimizer, verbose=False):
    with tf.GradientTape() as tape:
        loss_s_v, loss_neg_v, loss_c, loss_drop_c = model(C_ref_batch, training=True, verbose=verbose)
        loss = loss_s_v + loss_neg_v + loss_c + loss_drop_c  
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_s_v, loss_neg_v, loss_c, loss_drop_c

def train_model(
    model, 
    train_array, val_array=None,  # shape: (Z, (T+1)*k) or (Z, T_train*k) in 'forecast' mode
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    num_epochs=10, batch_size=10, patience=10,
    verbose=False,
    train_test_split='medium',
    x_fold=5
):
    """
    Trains the model, supporting 'medium' and 'forecast' split.
    """
    Z = train_array.shape[0]
    
    if train_test_split == 'medium':
        # Standard reshape as before
        C_train_all = train_array.reshape((Z, model.T, model.k))
        if val_array is not None:
            Z_val = val_array.shape[0]
            C_val_all = val_array.reshape((Z_val, model.T, model.k))
        else:
            C_val_all = None
    elif train_test_split == 'forecast':
        # Infer steps from shape and model.k
        T_train = train_array.shape[1] // model.k
        C_train_all = train_array.reshape((Z, T_train, model.k))
        if val_array is not None:
            Z_val = val_array.shape[0]
            T_val = val_array.shape[1] // model.k
            C_val_all = val_array.reshape((Z_val, T_val, model.k))
        else:
            C_val_all = None
    else:
        raise ValueError(f"Unknown train_test_split: {train_test_split}")

    # Track losses
    losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train = [], [], [], []
    losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val = [], [], [], []

    best_val_loss = np.inf
    best_weights = None
    steps_no_improv = 0
    n_batches = int(np.ceil(Z / batch_size))
    
    for epoch in range(num_epochs):
        idxs = np.arange(Z)
        np.random.shuffle(idxs)
        C_train_all = C_train_all[idxs]

        # Per-epoch sums
        epoch_loss_s_v, epoch_loss_neg_v, epoch_loss_c, epoch_loss_drop_c = 0.0, 0.0, 0.0, 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, Z)
            C_batch = C_train_all[start:end]
            # Pass the flag and x_fold to train_step
            loss_s_v, loss_neg_v, loss_c, loss_drop_c = train_step(
                model, C_batch, optimizer, verbose=verbose
            )
            epoch_loss_s_v += loss_s_v.numpy()
            epoch_loss_neg_v += loss_neg_v.numpy()
            epoch_loss_c += loss_c.numpy()
            epoch_loss_drop_c += loss_drop_c.numpy()

        # Average over batches
        epoch_loss_s_v /= n_batches
        epoch_loss_neg_v /= n_batches
        epoch_loss_c /= n_batches
        epoch_loss_drop_c /= n_batches

        losses_s_v_train.append(epoch_loss_s_v)
        losses_neg_v_train.append(epoch_loss_neg_v)
        losses_c_train.append(epoch_loss_c)
        losses_drop_c_train.append(epoch_loss_drop_c)

        if C_val_all is not None:
            val_s_v, val_neg_v, val_c, val_drop_c = model(C_val_all, training=False)
            val_s_v_np = val_s_v.numpy()
            val_neg_v_np = val_neg_v.numpy()
            val_c_np = val_c.numpy()
            val_drop_c_np = val_drop_c.numpy()
            losses_s_v_val.append(val_s_v_np)
            losses_neg_v_val.append(val_neg_v_np)
            losses_c_val.append(val_c_np)
            losses_drop_c_val.append(val_drop_c_np)

            val_loss = val_s_v_np + val_neg_v_np + val_c_np + val_drop_c_np
            if verbose:
                print(f'[Epoch {epoch+1}/{num_epochs}] '
                  f'Train: s_v={epoch_loss_s_v:.1e}, neg_v={epoch_loss_neg_v:.1e}, c={epoch_loss_c:.1e}, drop_c={epoch_loss_drop_c:.1e} | '
                  f'Val: s_v={val_s_v_np:.1e}, neg_v={val_neg_v_np:.1e}, c={val_c_np:.1e}, drop_c={val_drop_c_np:.1e}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                steps_no_improv = 0
            else:
                steps_no_improv += 1
                if steps_no_improv >= patience:
                    print('Early stopping triggered. Restoring best weights.')
                    model.set_weights(best_weights)
                    break
        else:
            if verbose:
                print(f'[Epoch {epoch+1}/{num_epochs}] '
                  f'Train: s_v={epoch_loss_s_v:.1e}, neg_v={epoch_loss_neg_v:.1e}, c={epoch_loss_c:.1e}, drop_c={epoch_loss_drop_c:.1e}')

    return (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
    )

###############################################################################
# PREDICT TIMECOURSE FOR VALIDATION 
###############################################################################

def predict_timecourse(model, C_ref_batch, verbose=False):
    """
    Predict the full timecourse of metabolite concentrations for a batch.
    Input shape: (batch_size, T+1, k)
    Returns: Tensor of shape (batch_size, T+1, k)
    """

    batch_size = tf.shape(C_ref_batch)[0]
    T = tf.shape(C_ref_batch)[1]
    C_pred = C_ref_batch[:, 0, :]
    lag_params = None
    predictions = tf.TensorArray(tf.float32, size=T)
    predictions = predictions.write(0, C_pred)
    for t in tf.range(T-1):
        ttf = tf.cast(t, tf.float32)
        C_next, v_t, lag_params = model._next_concentration(
        C_pred, ttf, training=False, lag_params=lag_params, 
        verbose=verbose
        )
        predictions = predictions.write(t+1, C_next)
        C_pred = C_next
    predictions_stacked = predictions.stack()
    predictions_stacked = tf.transpose(predictions_stacked, [1, 0, 2])
    return predictions_stacked

def predict_on_val_data(
    model,
    val_array,
    verbose=False
):
    """
    Predict the timecourse for the validation data, handling both 'medium' and 'forecast' split strategies.
    """
    Z_val = val_array.shape[0]
    ref = val_array.reshape((Z_val, model.T, model.k))
    pred = predict_timecourse(model, ref, verbose=verbose)
    return pred, ref

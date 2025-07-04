# %% [markdown]
# This version is a reactoring, weight for losses, introduction of loss on V < 0

# %% [markdown]
# ## Utilities

# %% [markdown]
# ### General

# %%
import cobra
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

import os
import json
# Conversion factor from OD to biomass (gDW/L)
ALPHA = 0.37

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###############################################################################
# GENERAL UTILITIES
###############################################################################

def area_between_curves(x, f, g):
    """
    Compute the area between two curves f(x) and g(x) using linear interpolation.
    
    Parameters:
        x: List or array of x-values (assumed sorted).
        f: List or array of f(x) values.
        g: List or array of g(x) values.
    
    Returns:
        Total area between the curves.
    """
    area = 0.0
    for i in range(len(x) - 1):
        # Linear interpolation slopes
        slope_f = (f[i+1] - f[i]) / (x[i+1] - x[i])
        slope_g = (g[i+1] - g[i]) / (x[i+1] - x[i])
        
        # Check for intersection within the interval
        if slope_f != slope_g:  # Avoid division by zero
            x_c = x[i] + (g[i] - f[i]) / (slope_f - slope_g)
            if x[i] < x_c < x[i+1]:  # Intersection lies within the interval
                # Compute areas for [x_i, x_c] and [x_c, x_{i+1}]
                f_c = f[i] + slope_f * (x_c - x[i])
                g_c = g[i] + slope_g * (x_c - x[i])
                area += 0.5 * abs(f[i] - g[i]) * (x_c - x[i])  # Area for [x_i, x_c]
                area += 0.5 * abs(f[i+1] - g[i+1]) * (x[i+1] - x_c)  # Area for [x_c, x_{i+1}]
                continue
        
        # No intersection, compute area directly
        diff1 = abs(f[i] - g[i])
        diff2 = abs(f[i+1] - g[i+1])
        area += 0.5 * (diff1 + diff2) * (x[i+1] - x[i])
    
    return area

def compute_curve_similarity(times, ref_OD, pred_OD):
    # Compute area btw the curves ref_OD and pred_OD
    curve_area = area_between_curves(times, ref_OD, pred_OD)    
    # Define the bounding box area
    x_min, x_max = np.min(times), np.max(times)
    y_min = min(np.min(pred_OD), np.min(ref_OD))
    y_max = max(np.max(pred_OD), np.max(ref_OD))
    total_area = (x_max - x_min) * (y_max - y_min)
    similarity = 1 - curve_area / total_area if total_area > 0 else 0
    return similarity

def ReLU(x):
    return x * (x > 0)


# %% [markdown]
# ### Plot

# %%
###############################################################################
# PLOTTING UTILS
###############################################################################

def plot_losses(title, loss_conc, loss_SV, num_epochs, save=''):
    """
    Generate a high-quality scientific plot comparing Loss_conc and Loss_SV.

    Parameters:
        title (str): Title of the plot.
        loss_conc (array-like): Loss concentration values.
        loss_SV (array-like): Loss SV values.
        num_epochs (int): Number of epochs.
        save: the folder where the curve has to be saved
    """
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)  # Increased dpi for high resolution

    # Update font styles for scientific publication
    plt.rcParams.update({
        'font.family': 'serif',  # Use serif fonts (e.g., Times New Roman)
        'font.size': 16,         # Standard font size
        'axes.labelsize': 18,    # Label size
        'axes.titlesize': 18,    # Title size
        'legend.fontsize': 14,   # Legend font size
        'xtick.labelsize': 14,   # X-axis tick labels
        'ytick.labelsize': 14,   # Y-axis tick labels
        'lines.linewidth': 2,    # Line thickness
        'lines.markersize': 6    # Marker size
    })

    # Primary Y-axis (left) for Loss_conc
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log(Loss OD)', color='blue')
    ax1.plot(range(1, len(loss_conc) + 1), loss_conc, label='Loss conc', color='blue')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create secondary Y-axis (right) for Loss_SV
    ax2 = ax1.twinx()
    ax2.set_ylabel('Log(Loss SV)', color='red')
    ax2.plot(range(1, len(loss_SV) + 1), loss_SV, label='Loss SV', color='red', linestyle='dashed')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title with extra padding to move it up
    plt.title(f'Loss OD & Loss SV over {num_epochs} Epochs with {title} set', pad=20)

    # Improve grid appearance
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Legends (ensure both axes' legends appear properly)
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save high-resolution version for publication
    title = title.replace(' ', '_')
    if save != '':
        plt.savefig(f'{save}/loss_plot_{title}.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_growth_curve(title, times, ref_OD, pred_OD, ref_std=None, pred_std=None, save=''):
    """
    Generate a high-quality scientific plot comparing reference OD and predicted OD.

    Parameters:
        title (str): Title of the plot.
        times (array-like): Time points.
        ref_OD (array-like): Reference optical density values.
        pred_OD (array-like): Predicted optical density values.
        ref_std (array-like, optional): Standard deviation for reference OD.
        pred_std (array-like, optional): Standard deviation for predicted OD.
        save: the folder where the curve has to be saved
    Returns:
        float: Similarity metric between the reference and predicted curves.
    """
    similarity = compute_curve_similarity(times, ref_OD, pred_OD)

    # Set font and figure size for scientific publication quality
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.family': 'serif',  # Use serif fonts (e.g., Times New Roman)
        'font.size': 16,         # Larger font size for readability
        'axes.labelsize': 18,    # Label size
        'axes.titlesize': 18,    # Title size
        'legend.fontsize': 14,   # Legend font size
        'xtick.labelsize': 14,   # X-axis tick labels
        'ytick.labelsize': 14,   # Y-axis tick labels
        'lines.linewidth': 2,    # Line thickness
        'lines.markersize': 6    # Marker size
    })

    # Define colors and markers for better distinction
    ref_color = 'black'
    pred_color = 'darkgreen'
    ref_marker = 'o'
    pred_marker = 's'
    
    # Plot reference OD (solid line)
    plt.plot(times, ref_OD, marker=ref_marker, linestyle='-', color=ref_color, label='Ref. OD')

    # Plot predicted OD (dashed line)
    plt.plot(times, pred_OD, marker=pred_marker, linestyle='--', color=pred_color, label='Pred. OD')

    # Plot shaded error regions for standard deviations
    if ref_std is not None:
        plt.fill_between(times, np.array(ref_OD) - np.array(ref_std), 
                         np.array(ref_OD) + np.array(ref_std), 
                         color=ref_color, alpha=0.2)
    
    if pred_std is not None:
        plt.fill_between(times, np.array(pred_OD) - np.array(pred_std), 
                         np.array(pred_OD) + np.array(pred_std), 
                         color=pred_color, alpha=0.2)

    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('log(OD)')
    plt.title(title, pad=20)

    # Add similarity annotation (no box)
    plt.text(0.05, 0.95, f"Similarity: {similarity:.2f}%", 
             transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top')

    # Improve grid appearance
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend with best placement
    plt.legend(frameon=True, loc='best', fancybox=True)

    # Ensure layout is tight for better spacing
    plt.tight_layout()

    # Save high-resolution version for publication
    if save != '':
        title = title.replace(' ', '_')
        plt.savefig(f'{save}/{title}.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return similarity

def plot_similarity_distribution(title, similarity_values, save=''):
    """
    Plots a histogram of similarity values with scientific publication quality.

    Parameters:
        similarity_values (array-like): 1D array of similarity values.
        title (str): Title of the plot.
        save: the folder where the curve has to be saved
    """

    # Create figure with high resolution
    plt.figure(figsize=(8, 6), dpi=300)

    # Update font styles for scientific publication
    plt.rcParams.update({
        'font.family': 'serif',  # Use serif fonts (e.g., Times New Roman)
        'font.size': 16,         # Standard font size
        'axes.labelsize': 18,    # Label size
        'axes.titlesize': 18,    # Title size
        'legend.fontsize': 14,   # Legend font size
        'xtick.labelsize': 14,   # X-axis tick labels
        'ytick.labelsize': 14,   # Y-axis tick labels
        'lines.linewidth': 2    # Line thickness
    })

    # Plot histogram
    plt.hist(similarity_values, bins='auto', color='grey', edgecolor='black', alpha=0.7)

    # Labels and title with padding for better spacing
    plt.xlabel('Similarity (%)')
    plt.ylabel('Frequency (%)')
    plt.title(title, pad=20)

    # Grid for readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save high-resolution figure
    if save != '':
        title = title.replace(' ', '_')
        plt.savefig(f'{save}/{title}.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()
 

# %% [markdown]
# ## Load data

# %%
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
        biomass_concentration = ALPHA * od_values  # resulting in gDW/L

        
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
    times: np.ndarray,
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
        row_flat = conc_matrix.reshape(-1)
        all_flat.append(row_flat)
    return np.stack(all_flat, axis=0)


# %% [markdown]
# ## Model

# %%
###############################################################################
# MODEL with BATCHED TIME-UNROLL and ACTUAL TIME UPDATES
###############################################################################

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

class FluxNetwork(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, hidden_layers=[100], dropout_rate=0.2, name='flux_network', **kwargs):
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
    def __init__(self, times, Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
                 hidden_layers=[100], dropout_rate=0.2, 
                 loss_weight_v = 1e-3,
                 loss_weight_decay_neg_v = 5e-1, 
                 loss_weight_decay_drop_conc  = 5e-1,
                 loss_weight_decay_conc = 5e-1,
                 verbose=True, name='metabolic_model'):
        super().__init__(name=name)
        self.times = times
        self.T = len(times) - 1  # T represents the number of intervals
        dt_vector = np.diff(times).astype(np.float32)
        dt_vector = dt_vector / np.min(dt_vector)
        self.dt_vector = tf.convert_to_tensor(dt_vector, dtype=tf.float32)
        self.Transport = Transport
        self.Stoichiometry = Stoichiometry
        self.k, self.n = Transport.shape[0], Transport.shape[1]
        self.rxn_ids = rxn_ids
        self.biomass_rxn_id = biomass_rxn_id
        self.biomass_flux_index = rxn_ids.index(biomass_rxn_id)
        self.hidden_layers = hidden_layers 
        self.dropout_rate = dropout_rate
        
        # weight to compute loss (for v and concentration)
        self.loss_weight_v = loss_weight_v
        self.loss_weight_decay_neg_v = loss_weight_decay_neg_v
        self.loss_weight_decay_drop_conc = loss_weight_decay_drop_conc
        self.loss_weight_decay_conc = loss_weight_decay_conc

        # Initialize FluxNetwork with a fixed name
        self.flux_net = FluxNetwork(self.k, self.n, hidden_layers, dropout_rate, name='flux_network')

    def build(self, input_shape):
        super().build(input_shape)
        # No additional build steps are necessary since all layers are initialized in __init__

    def printout(self):
        print(f'-----------------------------MetabolicModel-----------------------------')
        print(f'times: {self.times[0]:.2f}, {self.times[1]:.2f}, ..., {self.times[-1]:.2f}')
        print(f'T: {self.T}')
        print(f'dt: {self.dt_vector.numpy()}')
        print(f'Transport: {self.Transport.shape}')
        print(f'Stoichiometry: {self.Stoichiometry.shape}')
        print(f'n: {self.n}')
        print(f'k: {self.k}')
        print(f'Reaction ids: {len(self.rxn_ids)}')
        print(f'Biomass id: {self.biomass_rxn_id}')
        print(f'Biomass flux index: {self.biomass_flux_index}')
        print(f'Hidden Layers: {self.hidden_layers}')
        print(f'Dropout Rate: {self.dropout_rate}')
        print(f'------------------------------------------------------------------------')

    def call(self, C_ref_batch, training=False, verbose=False):
        """
        C_ref_batch: shape (batch_size, T+1, k)
        dt_vector: shape (T,) containing dt for each time interval.
        """
        batch_size = tf.shape(C_ref_batch)[0]
        T = tf.shape(C_ref_batch)[1] - 1  # T is number of intervals

        # Initial predictions from time 0.
        C_pred = C_ref_batch[:, 0, :]  # shape: (batch_size, k)
        total_loss_sv = 0.0
        total_loss_conc = 0.0

        for t in tf.range(T):

            # Compute fluxes; shape: (batch_size, n)
            v_t = self.flux_net(C_pred, training=training)

            # Compute change in external concentrations from fluxes; shape: (batch_size, k)
            delta_C_ext = tf.matmul(v_t, self.Transport, transpose_b=True)
            dt = self.dt_vector[t]  # scalar dt for this time interval
            C_next = C_pred + delta_C_ext * dt  # update using actual time difference

            # Reference concentrations for time t+1, shape: (batch_size, k)
            C_ref_next = C_ref_batch[:, t+1, :]

            # Compute loss on fluxes
            S_v = tf.matmul(v_t, self.Stoichiometry, transpose_b=True)  # shape: (batch_size, m)
            mse_sv = tf.reduce_mean(tf.square(S_v))
            # Add dynamic loss to penalize negative fluxes
            mse_neg_v = tf.reduce_mean(tf.nn.relu(-v_t))
            loss_weight_neg_v = tf.exp(-self.loss_weight_decay_neg_v * tf.cast(t, tf.float32)) 
            mse_sv +=  mse_neg_v * loss_weight_neg_v
            total_loss_sv += mse_sv * self.loss_weight_v  
        
            # Compute Loss on concentration
            mask = ~tf.math.is_nan(C_ref_next)
            mse_conc = tf.reduce_mean(tf.square(tf.boolean_mask(C_next, mask) - tf.boolean_mask(C_ref_next, mask)))
            # Add dynamic loss to penalize drop in biomass concentration
            mse_drop_conc = tf.reduce_mean(tf.nn.relu(C_pred[:,-1]-C_next[:,-1]))
            loss_weight_drop_conc = tf.exp(-self.loss_weight_decay_drop_conc * tf.cast(t, tf.float32)) 
            mse_conc += mse_drop_conc * loss_weight_drop_conc
            loss_weight_conc = tf.exp(-self.loss_weight_decay_conc * tf.cast(t, tf.float32)) 
            total_loss_conc += mse_conc * loss_weight_conc # * loss_weight_conc

            C_pred = C_next

            if verbose > 1:
                # print for first element in batch
                i = int(verbose)
                flux_biomass = v_t[self.biomass_flux_index]
                tf.print(f'Batch: {i} Time step {t} '
                         f'Glucose: {C_next[i, 0]:.3f} '
                         f'pred: {C_pred[i,-1]:.3f}  flux: {flux_biomass[i]:.3f} '
                         f'next: {C_next[i,-1]:.3f} ref-next: {C_ref_next[i, -1]:.3f} '
                         f'Loss SV: {total_loss_sv:.3f} Loss conc: {total_loss_conc:.3f}')

        return total_loss_sv, total_loss_conc

    def save_model(self, model_name='ODmodel', verbose=False):
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
        'T': int(self.T),
        'Transport': self.Transport.tolist(),
        'Stoichiometry': self.Stoichiometry.tolist(),
        'k': int(self.k),
        'n': int(self.n),
        'rxn_ids': self.rxn_ids,
        'biomass_rxn_id': self.biomass_rxn_id,
        'biomass_flux_index': int(self.biomass_flux_index),
        'hidden_layers': self.hidden_layers,
        'dropout_rate': float(self.dropout_rate)
        }
    
        # Write config to JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f)

        if verbose:
            print(f'Model weights saved to {weights_path} and config saved to {config_path}')

    @classmethod
    def load_model(cls, model_name='ODmodel', verbose=False):
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
        Transport = np.array(config['Transport'])
        Stoichiometry = np.array(config['Stoichiometry'])
        rxn_ids = config['rxn_ids']
        biomass_rxn_id = config['biomass_rxn_id']
        hidden_layers = config['hidden_layers']
        dropout_rate = config['dropout_rate']

        # Create a new model instance
        model = cls(
        times=times,
        Transport=Transport,
        Stoichiometry=Stoichiometry,
        rxn_ids=rxn_ids,
        biomass_rxn_id=biomass_rxn_id,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        verbose=verbose,
        name='metabolic_model'
        )

        # Ensure the model is built before loading weights
        dummy_input = tf.random.normal((1, len(times), model.k))  # Shape: (batch_size, T+1, k)
        _ = model(dummy_input)  # Forces all layers to initialize

        # Load saved weights
        model.load_weights(weights_path)

        if verbose:
            print(f'Model built from {config_path} and weights loaded from {weights_path}')

        return model


# %% [markdown]
# ## Training and evaluating

# %%
###############################################################################
# CREATE TRAIN AND CROSS-VALIDATE MODEL
###############################################################################

def create_model_train_val(
    media_file, od_file,
    cobra_model_file,
    biomass_rxn_id,
    x_fold=5, 
    hidden_layers=[100], dropout_rate=0.2,
    verbose=False
):
    # Load data and model matrices.
    times, metabolite_ids, experiment_data, dev_data, Stoichiometry, Transport, rxn_ids = process_data(
        media_file, od_file, cobra_model_file, biomass_rxn_id, verbose=verbose
    )
      
    # Split experiments into train and validation sets.
    exp_ids = list(experiment_data.keys())
    np.random.shuffle(exp_ids)
    val_ids = []
    if x_fold > 1:
        split = int(len(exp_ids) * (1 - 1/x_fold))
        train_ids = exp_ids[:split]
        val_ids = exp_ids[split:]
        val_data = {eid: experiment_data[eid] for eid in val_ids}
        # print('1',val_ids[0], ':', experiment_data[val_ids[0]]) # OK !!!
        # print('2', val_ids[0], ':', val_data[val_ids[0]]) # OK !!!
        val_array = prepare_experiment_array(times, metabolite_ids, val_data)

        # print('4', val_ids[0], ':', val_array[0]) # NOT OK
    else:
        train_ids = exp_ids
        val_array = None
        
    train_data = {eid: experiment_data[eid] for eid in train_ids}
    train_array = prepare_experiment_array(times, metabolite_ids, train_data)
    dev_data = np.asarray(list(dev_data.values()))
    train_dev = dev_data[:split]
    val_dev = dev_data[split:]

    if verbose:
        print(f'Train shape: {train_array.shape}')
        if val_array is not None:
            print(f'Val shape: {val_array.shape}')

    model = MetabolicModel(times, Transport, Stoichiometry, rxn_ids, biomass_rxn_id, 
            hidden_layers=hidden_layers, dropout_rate=dropout_rate, verbose=verbose)

    return model, train_array, train_dev, val_array, val_dev, val_ids
    
def train_step(model, C_ref_batch, optimizer, verbose=False):
    with tf.GradientTape() as tape:
        loss_sv, loss_conc = model(C_ref_batch, training=True, verbose=verbose)
        loss = loss_sv + loss_conc
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_sv, loss_conc

def evaluate_model_on_data(model, data_array):
    Z = data_array.shape[0]
    C_all = data_array.reshape((Z, model.T+1, model.k))
    loss_sv, loss_conc = model(C_all, training=False)
    return loss_sv.numpy(), loss_conc.numpy()

def train_model(
    model, 
    train_array, val_array=None,  # shape: (Z, (T+1)*k)
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    num_epochs=10, batch_size=10, patience=10,
    verbose=False
):
    Z = train_array.shape[0]
    C_train_all = train_array.reshape((Z, model.T+1, model.k))
    
    if val_array is not None:
        Z_val = val_array.shape[0]
        C_val_all = val_array.reshape((Z_val, model.T+1, model.k))
    else:
        C_val_all = None

    losses_sv_train, losses_conc_train = [], []
    losses_sv_val, losses_conc_val = [], []

    best_val_loss = np.inf
    best_weights = None
    steps_no_improv = 0
    n_batches = int(np.ceil(Z / batch_size))
    
    for epoch in range(num_epochs):
        # Shuffle experiments at each epoch
        idxs = np.arange(Z)
        np.random.shuffle(idxs)
        C_train_all = C_train_all[idxs]

        epoch_loss_sv, epoch_loss_conc = 0.0, 0.0
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, Z)
            C_batch = C_train_all[start:end]  # shape: (batch, T+1, k)
            loss_sv, loss_conc = train_step(model, C_batch, optimizer, verbose=verbose)
            epoch_loss_sv += loss_sv.numpy()
            epoch_loss_conc += loss_conc.numpy()

        epoch_loss_sv /= n_batches
        epoch_loss_conc /= n_batches
        losses_sv_train.append(epoch_loss_sv)
        losses_conc_train.append(epoch_loss_conc)

        if C_val_all is not None:
            val_sv, val_conc = model(C_val_all, training=False)
            val_sv_np = val_sv.numpy()
            val_conc_np = val_conc.numpy()
            losses_sv_val.append(val_sv_np)
            losses_conc_val.append(val_conc_np)
            val_loss = val_sv_np + val_conc_np
            print(f'[Epoch {epoch+1}/{num_epochs}] Train SV: {epoch_loss_sv:.1e} conc: {epoch_loss_conc:.1e} | '
                  f'Val SV: {val_sv_np:.1e} conc: {val_conc_np:.1e}')
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
            print(f'[Epoch {epoch+1}/{num_epochs}] Train SV: {epoch_loss_sv:.1e} Loss conc: {epoch_loss_conc:.1e}')

    return (losses_sv_train, losses_conc_train), (losses_sv_val, losses_conc_val)

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
    predictions = tf.TensorArray(tf.float32, size=T)
    C_pred = C_ref_batch[:, 0, :]  # (batch_size, T+1, k) -> (batch_size, k) 
    predictions = predictions.write(0, C_pred)
    for t in tf.range(T-1):
        v_t = model.flux_net(C_pred, training=False)
        delta_C_ext = tf.matmul(v_t, model.Transport, transpose_b=True)
        dt = model.dt_vector[t]
        C_next = C_pred + delta_C_ext * dt # TensorShape([56, 29]
        if verbose:
            for i in range(batch_size):
                flux_biomass = v_t[i, model.biomass_flux_index]
                tf.print(f'Batch: {i} Time step {t} Glucose: {C_next[i, 0]:.2f} '
                f'Ref: {C_ref_batch[i, t, -1]:.2f} Pred: {C_pred[i, -1]:.2f} '
                f'Flux: {flux_biomass:.2f} Next: {C_next[i, -1]:.2f}')
        predictions = predictions.write(t+1, C_next)
        C_pred = C_next
    predictions_stacked = predictions.stack()  # shape: (T+1, batch_size, k)
    predictions_stacked = tf.transpose(predictions_stacked, [1, 0, 2])
    return predictions_stacked

def predict_biomass_on_val_data(model, val_array, verbose=False):
    """
    Predict the biomass timecourse for the validation data.
    1. Reshape val_array (Z_val, (T+1)*k) to (Z_val, T+1, k)
    2. Obtain predicted biomass (last column) for each time point.
    
    Returns:
       pred_biomass:  predictions (Z_val, T+1) as a NumPy array
       ref_biomassD:  reference (Z_val, T+1) as a NumPy array
    """
    Z_val = val_array.shape[0]
    C_val_all = val_array.reshape((Z_val, model.T+1, model.k))
    predictions = predict_timecourse(model, C_val_all, verbose=verbose)
    # Extract biomass column (assumed last column)
    pred_biomass = predictions[:, :, -1].numpy()  # shape (Z_val, T+1) -- a tf.Tensor
    ref_biomass = C_val_all[:, :, -1]     # shape (Z_val, T+1) -- a NumPy array
    return pred_biomass, ref_biomass


# %% [markdown]
# # Main code

# %% [markdown]
# ### Train

# %%
###############################################################################
# TRAIN
###############################################################################

# Set environment (GPU configuration if needed)
print('Physical GPUs:', tf.config.list_physical_devices('GPU'))

# Update with your actual file names
folder = './'
run_name = 'Paul_OD_20_1.0'
media_file = folder+'data/'+'Paul_media.csv'                 
od_file = folder+'data/'+run_name+'.csv'               
cobra_model_file = folder+'data/'+'iML1515_duplicated.xml'
biomass_rxn_id = 'BIOMASS_Ec_iML1515_core_75p37M'

# Hyperparameters
seed = 10
np.random.seed(seed=seed)
hidden_layers = [500]
num_epochs = 1000
x_fold = 5       
batch_size = 10
patience = 100
N_iter = 3

# Split data
model, train_array, train_dev, val_array, val_dev, val_ids = create_model_train_val(
    media_file, od_file,
    cobra_model_file,
    biomass_rxn_id,
    x_fold=x_fold, 
    hidden_layers=hidden_layers, dropout_rate=0.2,
    verbose=False
)

# temporary saving
np.savetxt(f'{folder}model/{run_name}_val_array.txt', val_array, fmt='%f')
np.savetxt(f'{folder}model/{run_name}_val_dev.txt', val_dev, fmt='%f')
np.savetxt(f'{folder}model/{run_name}_val_ids.txt', np.asarray(val_ids), fmt='%d')

for i in range(N_iter):
    # Train
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=280,   # tune based on your experiments (280 experiments => one full cycle)
    decay_rate=0.9,
    staircase=True
    )
    (sv_train, conc_train), (sv_val, conc_val) = train_model(
    model, train_array, val_array=val_array,  
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    num_epochs=num_epochs, batch_size=batch_size, patience=patience,
    verbose=True
    )

    # save model
    model_name = f'{folder}model/{run_name}_{str(i)}'
    model.save_model(model_name=model_name, verbose=True)

    # Plot training and validation loss curves
    plot_losses('Training', conc_train, sv_train, num_epochs)
    if x_fold > 1:
        plot_losses('Validation', conc_val, sv_val, num_epochs)


# %% [markdown]
# ### Evaluate

# %%
###############################################################################
# TEST
###############################################################################

folder = './'
run_name = 'Paul_OD_20_1.0'
N_iter = 2
val_array = np.loadtxt(f'{folder}model/{run_name}_val_array.txt', dtype=float)
val_dev = np.loadtxt(f'{folder}model/{run_name}_val_dev.txt', dtype=float)
val_ids = np.loadtxt(f'{folder}model/{run_name}_val_ids.txt', dtype=int)
plot = True

# Predict OD and Plot
if val_array is not None:
    Pred, Ref = {}, {}
    for i in range(N_iter):
        model_name = f'{folder}model/{run_name}_{str(i)}'
        model = MetabolicModel.load_model(model_name=model_name, verbose=False)
        pred_val_od, ref_val_od = predict_biomass_on_val_data(model, val_array, verbose=False)
        pred = np.log(ReLU(pred_val_od / ALPHA) + 1.0e-8)
        ref  = np.log(ReLU(ref_val_od / ALPHA)  + 1.0e-8)
        Pred[i] = np.asarray(pred)
        Ref[i]  = np.asarray(ref)
    Ref, Pred = np.asarray(list(Ref.values())), np.asarray(list(Pred.values()))
    ref, pred = np.mean(Ref, axis=0), np.mean(Pred, axis=0)
    ref_std, pred_std = val_dev, np.std(Pred, axis=0) # val_pred is from input data file
    similarity = 0
    # For each experiment, compute an error metric and annotate the plot.
    Similarity = {}
    for i in range(ref.shape[0]):
        refi, predi = ref[i], pred[i]
        refi_std, predi_std = ref_std[i], pred_std[i]
        Similarity[i] = compute_curve_similarity(model.times, refi, predi)
        title = f'Experiment {int(val_ids[i])} Pred-True similarity: {Similarity[i]:.2f}'
        if i == 0:
            print(val_ids[i], ':', refi)
        if plot:
            plot_growth_curve(title, model.times, refi, predi, ref_std=refi_std, pred_std=predi_std)
    Similarity = np.asarray(list(Similarity.values()))
    
    print(f'Model: {model_name}  Pred-Ref similarity = {np.mean(Similarity):.2f}±{np.std(Similarity):.2f}')
    plot_similarity_distribution('Similarity Histogram', Similarity)


# %% [markdown]
# Parameter search
# 
#                  loss_weight_ng_v = 5e-1, 
#                  loss_weight_drop_conc  = 1e-1,
#                  loss_weight_decay_conc = 5e-1,
#                  loss_weight_sv = 1e-3,
# Model: ./model/Paul_OD_10_1.0_2 Average Pred-True similarity = 0.93
# 
#                  loss_weight_ng_v = 5e-1, 
#                  loss_weight_drop_conc  = 1e-1,
#                  loss_weight_decay_conc = 5e-1,
#                  loss_weight_sv = 1e-3,
# Model: ./model/Paul_OD_20_1.0_2  Pred-Ref similarity = 0.92±0.05
# 
# 
#                  loss_weight_v = 1e-3,
#                  loss_weight_decay_neg_v = 5e-1, 
#                  loss_weight_decay_drop_conc  = 5e-1,
#                  loss_weight_decay_conc = 5e-1,
# Model: ./model/Paul_OD_20_1.0_2  Pred-Ref similarity = 0.90±0.05
# 
# 
#                  loss_weight_v = 1e-3,
#                  loss_weight_decay_neg_v = 5e-1, 
#                  loss_weight_decay_drop_conc  = 5e-1,
#                  loss_weight_decay_conc = 5e-1,
# Model: ./model/Paul_OD_20_1.0_2  Pred-Ref similarity = 0.93±0.05
# 
#                  loss_weight_v = 1e-3,
#                  loss_weight_decay_neg_v = 5e-1, 
#                  loss_weight_decay_drop_conc  = 5e-1,
#                  loss_weight_decay_conc = 5e-1,
# Model: ./model/Paul_OD_20_1.0_2  Pred-Ref similarity = 0.93±0.05
# 
#                  loss_weight_v = 1e-3,
#                  loss_weight_decay_neg_v = 5e-1, 
#                  loss_weight_decay_drop_conc  = 5e-1,
#                  loss_weight_decay_conc = 5e-1,
# Model: ./model/Paul_OD_20_1.0_1  Pred-Ref similarity = 0.93±0.05

# %%




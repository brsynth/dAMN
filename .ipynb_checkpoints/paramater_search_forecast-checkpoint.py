import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import numpy as np
import tensorflow as tf
import random
from itertools import product
import json
import pandas as pd

utils.OD_TO_CONC = 0.37

# Parameter search spaces SV, neg V, C, drop C
l = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
k = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
l1, l2, l3 = l.copy(), l.copy(), l.copy()
k1, k2, k3 = k.copy(), k.copy(), k.copy()
N_search = 50  # Reduce for reasonable runtime, increase for more exhaustive search
param_grid = list(product(l1, l2, l3, k1, k2, k3))
random.seed(1)
param_samples = random.sample(param_grid, min(N_search, len(param_grid)))
short_num_epochs = 50
batch_size = 10
patience = 30
x_fold = 3

folder = './'
run_name = 'TUNE_PINN'
media_file = folder+'data/'+'Paul_media.csv'
od_file = folder+'data/'+'Paul_OD_20_1.0.csv'
cobra_model_file = folder+'data/'+'iML1515_duplicated.xml'
biomass_rxn_id = 'BIOMASS_Ec_iML1515_core_75p37M'
hidden_layers_lag = [56]
hidden_layers_flux = [460]

train_test_split = 'forecast'

# --- Load all reference and metadata (file order) ---
media_df = pd.read_csv(media_file)
exp_ids_in_file = list(media_df['ID'])
times, metabolite_ids, experiment_data, dev_data, Stoichiometry, Transport, rxn_ids = utils.process_data(
    media_file, od_file, cobra_model_file, biomass_rxn_id
)

full_ref_matrix = np.stack([experiment_data[exp_id] for exp_id in exp_ids_in_file], axis=0)  # (n_exp, n_times, k)
n_exp, n_times, k_met = full_ref_matrix.shape
forecast_train_steps = int(round(n_times * (x_fold - 1) / x_fold))

results = []

for idx, (l1_, l2_, l3_, k1_, k2_, k3_) in enumerate(param_samples):
    print(f"\n*** Testing λ={[l1_, l2_, l3_, 0]}, k={[k1_, k2_, k3_, 0]} ({idx+1}/{len(param_samples)}) ***")

    # --- Prepare batch input for training (first X-1 folds for all experiments) ---
    train_array = np.stack(
        [full_ref_matrix[i, :forecast_train_steps, :].flatten() for i in range(n_exp)],
        axis=0
    )

    # --- Model creation ---
    model = utils.MetabolicModel(
        times, Transport, Stoichiometry, rxn_ids, biomass_rxn_id,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        dropout_rate=0.2,
        loss_weight=[l1_, l2_, l3_, 0],
        loss_decay=[k1_, k2_, k3_, 0],
        verbose=False
    )
    
    # --- Training ---
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=280, decay_rate=0.9, staircase=True
    )
    (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        _
    ) = utils.train_model(
        model, train_array, val_array=None,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        num_epochs=short_num_epochs, batch_size=batch_size, patience=patience,
        verbose=True,
        train_test_split=train_test_split,
        x_fold=x_fold
    )

    # --- Save model ---
    model_name = f'{folder}model/{run_name}_search_forecast_{idx}'
    model.save_model(model_name=model_name, verbose=False)
    model = utils.MetabolicModel.load_model(model_name=model_name, verbose=False)
    target_exp_ids = list(media_df['ID'])
    target_indices = [i for i, exp_id in enumerate(exp_ids_in_file) if exp_id in target_exp_ids]

    # --- Prepare batch input for only those experiments ---
    val_array = np.stack(
        [full_ref_matrix[i, 0, :] for i in target_indices],
        axis=0
    ) 

    pred_biomass, _ = utils.predict_on_val_data(
        model, val_array, verbose=False,
        train_test_split='forecast',
        forecast_total_steps=n_times,
        forecast_train_steps=forecast_train_steps
    )
    
    # --- Convert to log(OD) for forecast region only ---
    biomass_col_idx = metabolite_ids.index('BIOMASS')
    Ref_biomass = full_ref_matrix[:, :, biomass_col_idx]
    pred_logOD = np.log(np.maximum(pred_biomass / utils.OD_TO_CONC, 1e-8))
    ref_logOD  = np.log(np.maximum(Ref_biomass / utils.OD_TO_CONC, 1e-8))

    Similarity = []
    forecast_idx = list(range(forecast_train_steps, n_times))
    time_axis = times[forecast_idx]
    for i in range(n_exp):
        ref_curve = ref_logOD[i, forecast_idx]
        pred_curve = pred_logOD[i, forecast_idx]
        Similarity.append(utils.compute_curve_similarity(time_axis, ref_curve, pred_curve))
    Similarity = np.asarray(Similarity)
    mean_sim = np.mean(Similarity)
    std_sim = np.std(Similarity)
    print(f"Mean test similarity (forecast region only): {mean_sim:.3f} ± {std_sim:.3f}")

    results.append({
        "l1": l1_,
        "l2": l2_,
        "l3": l3_,
        "l4": 0,
        "k1": k1_,
        "k2": k2_,
        "k3": k3_,
        "k4": 0,
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "model_name": model_name,
    })

# --- Results output ---
results = sorted(results, key=lambda x: x["mean_similarity"], reverse=True)
print("\n*** Parameter Search Results (top 5): ***")
for res in results[:5]:
    print(f"λ={[res['l1'], res['l2'], res['l3'], res['l4']]}, k={[res['k1'], res['k2'], res['k3'], res['k4']]} => Mean similarity: {res['mean_similarity']:.3f} ± {res['std_similarity']:.3f}")

best = results[0]
print(f"\nBest parameters for full run: λ={[best['l1'], best['l2'], best['l3'], best['l4']]}, k={[best['k1'], best['k2'], best['k3'], best['k4']]}")
with open(f'{folder}model/hyperparameter_search_results_forecast.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nFull Table: Hyperparameter Search Results")
print("l1\tl2\tl3\tl4\tk1\tk2\tk3\tk4\tMean similarity\tStd similarity\tModel Name")
for res in results:
    print(f"{res['l1']}\t{res['l2']}\t{res['l3']}\t{res['l4']}\t{res['k1']}\t{res['k2']}\t{res['k3']}\t{res['k4']}\t{res['mean_similarity']:.3f}\t{res['std_similarity']:.3f}\t{res['model_name']}")

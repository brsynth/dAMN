import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import numpy as np
import tensorflow as tf
import random
from itertools import product
import json

from sklearn.metrics import r2_score

# Fixed config
folder = './'
media_file = folder + 'data/' + 'M28_media.csv'
od_file = folder + 'data/' + 'M28_OD_20.csv'
cobra_model_file = folder + 'data/' + 'iML1515_duplicated.xml'
biomass_rxn_id = 'BIOMASS_Ec_iML1515_core_75p37M'
hidden_layers_lag = [50]
hidden_layers_flux = [500]

os.makedirs("results_params", exist_ok=True)


# Parameter search spaces
l = [0.0001, 0.001, 0.01, 0.1, 1.0, 1, 2]
k = [0.0, 0.25, 0.5, 0.75, 1.0]
l1, l2, l3, l4 = l.copy(), l.copy(), l.copy(), l.copy() 
k1, k2, k3, k4 = k.copy(), k.copy(), k.copy(), k.copy()
N_search = 25
param_grid = list(product(l1, k2, k3, k4))
random.seed(1)
param_samples = random.sample(param_grid, min(N_search, len(param_grid)))

# Training config
short_num_epochs = 500
batch_size = 10
patience = 30
x_fold = 3
k1_ = 0

for mode in ['forecast', 'medium']:
    print(f"\n=== Running mode: {mode} ===\n")
    results = []

    for idx, (l1_, k2_, k3_, k4_) in enumerate(param_samples):
        # Model + data split
        model, train_array, train_dev, val_array, val_dev, val_ids = utils.create_model_train_val(
            media_file, od_file, cobra_model_file, biomass_rxn_id,
            x_fold=x_fold,
            hidden_layers_lag=hidden_layers_lag,
            hidden_layers_flux=hidden_layers_flux,
            dropout_rate=0.2,
            loss_weight =[l1_, 1, 1, 1],
            loss_decay  =[k1_, k2_, k3_, k4_],
            verbose=False,
            train_test_split=mode
        )

        # Training
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=280, decay_rate=0.9, staircase=True
        )
        (
            (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
            (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
        ) = utils.train_model(
            model, train_array, val_array=val_array,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            num_epochs=short_num_epochs, batch_size=batch_size, patience=patience,
            verbose=False,
            train_test_split=mode,
            x_fold=x_fold
        )

        # Evaluate
        pred, ref = utils.predict_on_val_data(model, val_array, verbose=False)
        pred, ref = utils.concentration_to_OD(pred), utils.concentration_to_OD(ref)
        pred = np.expand_dims(pred, axis=0)
        ref = np.expand_dims(ref, axis=0)
        R2 = utils.r2_growth_curve(pred, ref, OD=True)
        R2_mean, R2_std = np.mean(R2), np.std(R2)

        print(f'[{mode.upper()}] {idx+1}/{len(param_samples)}: λ={[l1_, 1, 1, 1]}, k={[k1_, k2_, k3_, k4_]} -> R²: {R2_mean:.3f} ± {R2_std:.3f}')

        # Store result
        results.append({
            "l1": l1_,
            "l2": 1,
            "l3": 1,
            "l4": 1,
            "k1": k1_,
            "k2": k2_,
            "k3": k3_,
            "k4": k4_,
            "R2_mean": R2_mean,
            "R2_std":  R2_std
        })

    # Sort and save results
    results = sorted(results, key=lambda x: x["R2_mean"], reverse=True)
    out_file = f'results/results_{mode}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Saved results to {out_file}\n")
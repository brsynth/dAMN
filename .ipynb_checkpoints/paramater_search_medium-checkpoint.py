import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import numpy as np
import tensorflow as tf
import random
from itertools import product
import json

# Parameter search spaces
l = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
k = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
l1 = l.copy()
k2, k3, k4 = k.copy(), k.copy(), k.copy()
N_search = 30
param_grid = list(product(l1, k2, k3, k4))
random.seed(1)
param_samples = random.sample(param_grid, min(N_search, len(param_grid)))

short_num_epochs = 100
batch_size = 10
patience = 30
x_fold = 5

folder = './'
media_file = folder+'data/'+'Paul_media.csv'
od_file = folder+'data/'+'Paul_OD_20_1.0.csv'
cobra_model_file = folder+'data/'+'iML1515_duplicated.xml'
biomass_rxn_id = 'BIOMASS_Ec_iML1515_core_75p37M'
hidden_layers_lag = [56]
hidden_layers_flux = [460]

train_test_split = 'medium'

results = []

for idx, (l1_, k2_, k3_, k4_) in enumerate(param_samples):
    print(f"\n*** Testing λ={[l1_, 1, 1, 1]}, k={[0, k2_, k3_, k4_]} ({idx+1}/{len(param_samples)}) ***")
    model, train_array, train_dev, val_array, val_dev, val_ids = utils.create_model_train_val(
        media_file, od_file, cobra_model_file, biomass_rxn_id,
        x_fold=x_fold,
        hidden_layers_lag=hidden_layers_lag,
        hidden_layers_flux=hidden_layers_flux,
        dropout_rate=0.2,
        loss_weight=[l1_, 1, 1, 1],
        loss_decay=[0, k2_, k3_, k4_],
        verbose=False,
        train_test_split=train_test_split
    )
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
        train_test_split=train_test_split,
        x_fold=x_fold
    )
    #model_name = f'{folder}model/{run_name}_search_medium_{idx}'
    #model.save_model(model_name=model_name, verbose=False)
    #model = utils.MetabolicModel.load_model(model_name=model_name, verbose=False)
    pred, ref = utils.predict_on_val_data(
        model, val_array, verbose=False, train_test_split=train_test_split
    )
    Similarity = []
    time_axis = model.times
    for i in range(pred.shape[0]):
        Similarity.append(utils.compute_curve_similarity(time_axis, ref[i], pred[i]), OD=True)
    Similarity = np.asarray(Similarity)
    mean_sim = np.mean(Similarity)
    std_sim = np.std(Similarity)
    print(f"Mean test similarity: {mean_sim:.3f} ± {std_sim:.3f}")

    # Record
    results.append({
        "l1": l1_,
        "l2": 1,
        "l3": 1,
        "l4": 1,
        "k1": 0,
        "k2": k2_,
        "k3": k3_,
        "k4": k4_,
        "mean_similarity": mean_sim,
        "std_similarity": std_sim,
        "model_name": model_name,
    })

results = sorted(results, key=lambda x: x["mean_similarity"], reverse=True)
print("\n*** Parameter Search Results (top 5): ***")
for res in results[:5]:
    print(f"λ={[res['l1'], res['l2'], res['l3'], res['l4']]}, k={[res['k1'], res['k2'], res['k3'], res['k4']]} => Mean similarity: {res['mean_similarity']:.3f} ± {res['std_similarity']:.3f}")

best = results[0]
print(f"\nBest parameters for full run: λ={[best['l1'], best['l2'], best['l3'], best['l4']]}, k={[best['k1'], best['k2'], best['k3'], best['k4']]}")
with open(f'{folder}model/hyperparameter_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nFull Table: Hyperparameter Search Results")
print("l1\tl2\tl3\tl4\tk1\tk2\tk3\tk4\tMean similarity\tStd similarity\tModel Name")
for res in results:
    print(f"{res['l1']}\t{res['l2']}\t{res['l3']}\t{res['l4']}\t{res['k1']}\t{res['k2']}\t{res['k3']}\t{res['k4']}\t{res['mean_similarity']:.3f}\t{res['std_similarity']:.3f}\t{res['model_name']}")

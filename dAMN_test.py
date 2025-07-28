# Test

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sklearn
import utils
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

print('Physical GPUs:', tf.config.list_physical_devices('GPU'))

metabolite_ids = [
'glc__D_e', 'xyl__D_e', 'succ_e', 'ala__L_e', 'arg__L_e', 'asn__L_e', 'asp__L_e',
'cys__L_e', 'glu__L_e', 'gln__L_e', 'gly_e', 'his__L_e', 'ile__L_e', 'leu__L_e',
'lys__L_e', 'met__L_e', 'phe__L_e', 'pro__L_e', 'ser__L_e', 'thr__L_e', 'trp__L_e',
'tyr__L_e', 'val__L_e', 'ade_e', 'gua_e', 'csn_e', 'ura_e', 'thymd_e', 'BIOMASS'
]

train_test_split = 'forecast' # 'forecast' or 'medium'
folder = './'
file_name = 'M28_OD_20'
N_iter = 1
run_name = f'{file_name}_{train_test_split}'
OD = True # when True biomass concentration transformed in OD
plot =  'substrate' # 'growth' or 'substrate'

# Load
val_array = np.loadtxt(f'{folder}model/{run_name}_val_array.txt', dtype=float)
val_dev = np.loadtxt(f'{folder}model/{run_name}_val_dev.txt', dtype=float)
val_ids = np.loadtxt(f'{folder}model/{run_name}_val_ids.txt', dtype=int)
if val_array is None:
    raise ValueError(f'Validation file not found: {folder}model/{run_name}_val_array.txt')

# Predict
Pred, Ref, Pred_bio, Ref_bio = {}, {}, {}, {}
for i in range(N_iter):
    model_name = f'{folder}model/{run_name}_{str(i)}'
    model = utils.MetabolicModel.load_model(model_name=model_name, verbose=False)
    model.metabolite_ids = metabolite_ids if len(model.metabolite_ids) == 0 else model.metabolite_ids
    pred, ref = utils.predict_on_val_data(model, val_array, verbose=False) # 1, 86, 157
    pred, ref = np.asarray(pred), np.asarray(ref)
    Pred[i], Ref[i] = pred, ref
Pred , Ref = np.asarray(list(Pred.values())), np.asarray(list(Ref.values()))
R2 = utils.r2_growth_curve(Pred, Ref, OD=OD)

print(f'Model: {run_name}  R2 = {np.mean(R2):.2f}±{np.std(R2):.2f} Median = {np.median(R2):.2f}')
title = f"R2 Histogram {train_test_split}"
utils.plot_similarity_distribution(title, R2, save="./figure")

# Plot
if plot == 'growth':
    utils.plot_predicted_reference_growth_curve(
    times=model.times,
    Pred=Pred, Ref=Ref, val_dev=val_dev,
    OD=OD,R2=R2,
    train_time_steps=model.train_time_steps if hasattr(model, "train_time_steps") else 0,
    experiment_ids=list(val_ids),
    run_name=run_name,
    train_test_split=train_test_split,
    save="./figure"
    )
elif plot == 'substrate':
    utils.plot_predicted_biomass_and_substrate(
    model.times, Pred,
    experiment_ids=list(val_ids),
    metabolite_ids=list(model.metabolite_ids),
    run_name=run_name,
    train_test_split=train_test_split,
    save="./figure"
    )
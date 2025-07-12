import utils # AMN defined functions
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import os

# env and var etup
# Conversion factor from OD to biomass (gDW/L)
utils.ALPHA = 0.37
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        model = utils.MetabolicModel.load_model(model_name=model_name, verbose=False)
        pred_val_od, ref_val_od = utils.predict_biomass_on_val_data(model, val_array, verbose=False)
        pred = np.log(utils.ReLU(pred_val_od / utils.ALPHA) + 1.0e-8)
        ref  = np.log(utils.ReLU(ref_val_od / utils.ALPHA)  + 1.0e-8)
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
        Similarity[i] = utils.compute_curve_similarity(model.times, refi, predi)
        title = f'Experiment {int(val_ids[i])} Pred-True similarity: {Similarity[i]:.2f}'
        if i == 0:
            print(val_ids[i], ':', refi)
        if plot:
            utils.plot_growth_curve(title, model.times, refi, predi, ref_std=refi_std, pred_std=predi_std, save="./figure")
    Similarity = np.asarray(list(Similarity.values()))
    
    print(f'Model: {model_name}  Pred-Ref similarity = {np.mean(Similarity):.2f}±{np.std(Similarity):.2f}')
    utils.plot_similarity_distribution('Similarity Histogram', Similarity)
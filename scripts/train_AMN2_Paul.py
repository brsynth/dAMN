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

# reduce model
reduce = True # Set at True if you want to reduce the model

# Hyperparameters
seed = 10
np.random.seed(seed=seed)
hidden_layers = [500]
num_epochs = 3 # 1000
x_fold = 5       
batch_size = 10
patience = 100
N_iter = 3

# Split data
model, train_array, train_dev, val_array, val_dev, val_ids = utils.create_model_train_val(
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
    (sv_train, conc_train), (sv_val, conc_val) = utils.train_model(
    model, train_array, val_array=val_array,  
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    num_epochs=num_epochs, batch_size=batch_size, patience=patience,
    verbose=True
    )

    # save model
    model_name = f'{folder}model/{run_name}_{str(i)}'
    model.save_model(model_name=model_name, verbose=True)

    # Plot training and validation loss curves
    utils.plot_losses('Training', conc_train, sv_train, num_epochs, save="./figure")

    if x_fold > 1:
        utils.plot_losses('Validation', conc_val, sv_val, num_epochs, save="./figure")



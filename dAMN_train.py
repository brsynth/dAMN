import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import numpy as np
import tensorflow as tf
print('Physical GPUs:', tf.config.list_physical_devices('GPU'))

train_test_split = 'forecast' # 'forecast' or 'medium'
folder = './'
file_name = 'M28_OD_20'
media_file = folder+'data/'+'M28_media.csv'                 
od_file = folder+'data/'+file_name+'.csv'               
cobra_model_file = folder+'data/'+'iML1515_duplicated.xml'
biomass_rxn_id = 'BIOMASS_Ec_iML1515_core_75p37M'

# Hyperparameters
seed = 10
np.random.seed(seed=seed)
hidden_layers_lag = [50]
hidden_layers_flux = [500]
num_epochs = 1000
x_fold = 3      
batch_size = 10
patience = 100
N_iter = 3
run_name = f'{file_name}_{train_test_split}'

# Create model
model, train_array, train_dev, val_array, val_dev, val_ids = utils.create_model_train_val(
    media_file, od_file,
    cobra_model_file,
    biomass_rxn_id,
    x_fold=x_fold, 
    hidden_layers_lag=hidden_layers_lag, 
    hidden_layers_flux=hidden_layers_flux, 
    dropout_rate=0.2,
    loss_weight=[0.001, 1, 1, 1], 
    loss_decay=[0, 0.5, 0.5, 1], 
    verbose=True,
    train_test_split=train_test_split
)

# Saving for future testing and validation
np.savetxt(f'{folder}model/{run_name}_val_array.txt', val_array, fmt='%f')
np.savetxt(f'{folder}model/{run_name}_val_dev.txt', val_dev, fmt='%f')
np.savetxt(f'{folder}model/{run_name}_val_ids.txt', np.asarray(val_ids), fmt='%d')

# Train model
for i in range(N_iter):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=280,
        decay_rate=0.9,
        staircase=True
    )
    (
        (losses_s_v_train, losses_neg_v_train, losses_c_train, losses_drop_c_train),
        (losses_s_v_val, losses_neg_v_val, losses_c_val, losses_drop_c_val)
    ) = utils.train_model(
        model, train_array, val_array=val_array,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        num_epochs=num_epochs, batch_size=batch_size, patience=patience,
        verbose=True,
        train_test_split=train_test_split,
        x_fold=x_fold
    )

    model_name = f'{folder}model/{run_name}_{str(i)}'
    model.save_model(model_name=model_name, verbose=True)

    utils.plot_loss('Training S_v', losses_s_v_train, num_epochs, save="./figure")
    utils.plot_loss('Training Neg_v', losses_neg_v_train, num_epochs, save="./figure")
    utils.plot_loss('Training C', losses_c_train, num_epochs, save="./figure")
    utils.plot_loss('Training Drop_c', losses_drop_c_train, num_epochs, save="./figure")

    if x_fold > 1:
        utils.plot_loss('Validation S_v', losses_s_v_val, num_epochs, save="./figure")
        utils.plot_loss('Validation Neg_v', losses_neg_v_val, num_epochs, save="./figure")
        utils.plot_loss('Validation C', losses_c_val, num_epochs, save="./figure")
        utils.plot_loss('Validation Drop_c', losses_drop_c_val, num_epochs, save="./figure")

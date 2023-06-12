import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import glob
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import ast
from tensorflow.keras.layers import MultiHeadAttention
import json
import time
import itertools
import random

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

job_id = sys.argv[1]
tune_index = int(sys.argv[2])

model_type = 'cnnFZ'
run = 'runBB'
cur_dir = f'/tmp/Abhinav_DATA{job_id}/' 
best_model_name = f'{cur_dir}models/fz_predict_{model_type}_{job_id}_'
root_dir = '/hercules/scratch/atya/BinaryML/'
log_dir = f'hyperparameter_tuning/{model_type}/'
cur_log_dir = os.path.join(cur_dir,log_dir)
list_of_dicts_dir = os.path.join(cur_log_dir,'list_of_dicts.json')

myexecute(f'mkdir -p {cur_dir}raw_data/{run}/')
myexecute(f'mkdir -p {cur_dir}models/')
myexecute(f'mkdir -p {cur_log_dir}')
myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Binary simulations predictor: Predicting the frequency and drift. \nBest model is under:{best_model_name} \
           \n\n\n ############################################################################## \n\n\n \"')

#files to sync
files1 = glob.glob(f'{root_dir}raw_data/{run}/*chunk.npy')
for file in files1:
    myexecute(f'rsync -Pav -q {file} {cur_dir}raw_data/{run}/')

myexecute(f'rsync -Pav -q {root_dir+f"hyperparameter_tuning/{model_type}/*"} {cur_log_dir}')

X_train = np.load(cur_dir + f'raw_data/{run}/train_data_chunk.npy').astype(np.float64)
X_test = np.load(cur_dir + f'raw_data/{run}/test_data_chunk.npy').astype(np.float64)
X_val = np.load(cur_dir + f'raw_data/{run}/val_data_chunk.npy').astype(np.float64)
X_train = X_train/np.max(X_train,axis=1)[:,None]
X_test = X_test/np.max(X_test,axis=1)[:,None]
X_val = X_val/np.max(X_val,axis=1)[:,None]

Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_chunk.npy').astype(np.float64)
Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_chunk.npy').astype(np.float64)
Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_chunk.npy').astype(np.float64)

Y_train[:,1] = 8*np.abs(Y_train[:,1])
Y_test[:,1] = 8*np.abs(Y_test[:,1])
Y_val[:,1] = 8*np.abs(Y_val[:,1])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

Y_train = (Y_train.reshape(Y_train.shape[0],Y_train.shape[1], 1))
Y_val = (Y_val.reshape(Y_val.shape[0],Y_val.shape[1], 1))
Y_test = (Y_test.reshape(Y_test.shape[0],Y_test.shape[1], 1))

# Create data generators
def create_dataset(X, y, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def period_accuracy(Y_test, Y_pred):
    """
    Computes the percentage of predicted periods that are within 0.5% of the true periods.
    """
    accuracies = []
    Y_test = tf.cast(Y_test, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    shape = Y_test.shape[1]
    for i in range(shape):
        rat = (Y_test[:,i]+ 1e-10)/(Y_pred[:,i]+ 1e-10)
        rat_bool = tf.logical_and(tf.less(rat, 1.005), tf.greater(rat, 0.995))
        accuracy = tf.reduce_mean(tf.cast(rat_bool, tf.float32)) * 100
        accuracies.append(accuracy)
    return [float(acc.numpy()) for acc in accuracies]


def median_percent_deviation(Y_test, Y_pred):
    """
    Computes the median percent deviation from the predicted period.
    """
    Y_test = tf.cast(Y_test, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    median_deviations = []
    shape = Y_test.shape[1]
    for i in range(shape):
        
        Y_test_i = Y_test[:,i]
        Y_pred_i = Y_pred[:,i]

        percent_deviation = tf.abs((Y_test_i - Y_pred_i) / (Y_test_i + 1e-10)) * 100
        median_deviation = tfp.stats.percentile(percent_deviation, q=50)
        median_deviations.append(median_deviation)

    return [float(med.numpy()) for med in median_deviations]


#check if the list of dictionaries already exists
if os.path.exists(list_of_dicts_dir):
    list_of_dicts = json.load(open(list_of_dicts_dir))
    myexecute(f'echo "Loaded list of dictionaries from {list_of_dicts_dir}"')

else:
    myexecute(f'mkdir -p {root_dir}hyperparameter_tuning/{model_type}/')
    myexecute(f'echo "List of dictionaries not found, creating all search combinations"')
    default_hyperparameters = {
        'num_cnn_layers': 4,
        'num_deep_layers': 3,
        'initial_learning_rate': 0.0005,
        'decay_rate': 0.9,
        'batch_size': 1200,
        'index':0,
        'deep_layer_size':str([64,128,256]),
        'epochs':20000,
        'patience':400,
        'input_shape':(400,1),
        'padding':'same',
        'dilation':False,
        'conv1d_filters':str([64,128,256,512]),
        'conv1d_kernel_size':str([5,7,7,7]),
        'batch_normalization':False,
    }

    search_space = {
        'num_deep_layers': [2,4,6,8,10,12],
        'initial_learning_rate': [round(i, 5) for i in np.logspace(-4, -2, 50)],
        'decay_rate': [round(i, 2) for i in np.arange(0.5, 1.0, 0.05)],
        'batch_size': [400,800,1200],
        'num_cnn_layers': [2,4,6,8,10,12],
        'padding':['same','valid'],
        'dilation':[True,False],
        'batch_normalization':[True,False],
    }
    
    start_time = time.time()
    # Generate all possible combinations of hyperparameters
    all_combinations = list(itertools.product(*search_space.values()))

    search_space_keys = list(search_space.keys())

    list_of_dicts = [{search_space_keys[i]: combination[i] for i in range(len(search_space_keys))} for combination in all_combinations]
    random.Random(4).shuffle(list_of_dicts)

    index = 1
    new_list_of_dicts = []
    new_list_of_dicts.append(default_hyperparameters)
    for i,ele in enumerate(list_of_dicts[:9999]): # set maimum number of trials that can be tested to 10000
        list_deep_layer_size = []
        list_dilation_rate_size = []
        list_conv1d_filters = []
        list_conv1d_kernel_size = []    
        for j in range(150):
            
            if j < 50:
                start = random.sample(range(8, 257, 8), 1)[0]
                list_deep_layer_size.append(str([start*i for i in range(1,ele['num_deep_layers']+1)]))
                if ele['dilation']:
                    start = random.sample(range(2, 17, 2), 1)[0]
                    list_dilation_rate_size.append(str([start*i for i in range(1,ele['num_cnn_layers']+1)]))
                start = random.sample(range(8, 257, 8), 1)[0]
                list_conv1d_filters.append(str([start*i for i in range(1,ele['num_cnn_layers']+1)]))
                start = random.sample(range(2,8,1),1)[0]
                list_conv1d_kernel_size.append(str([start*i for i in range(1,ele['num_cnn_layers']+1)]))
            elif (j>=50) & (j < 100):
                end = random.sample(range(8, 257, 8), 1)[0]
                list_deep_layer_size.append(str([end*i for i in range(ele['num_deep_layers']+1,1,-1)]))
                if ele['dilation']:
                    end = random.sample(range(2, 17, 2), 1)[0]
                    list_dilation_rate_size.append(str([end*i for i in range(ele['num_cnn_layers']+1,1,-1)]))
                end = random.sample(range(8, 257, 8), 1)[0]
                list_conv1d_filters.append(str([end*i for i in range(ele['num_cnn_layers']+1,1,-1)]))
                end = random.sample(range(2,8,1),1)[0]
                list_conv1d_kernel_size.append(str([end*i for i in range(ele['num_cnn_layers']+1,1,-1)]))
            else:
                list_deep_layer_size.append(str(random.sample(range(8, 257*ele['num_deep_layers'], 8), ele['num_deep_layers'])))
                if ele['dilation']:
                    list_dilation_rate_size.append(str(random.sample(range(2, 17*ele['num_cnn_layers'], 2), ele['num_cnn_layers'])))
                list_conv1d_filters.append(str(random.sample(range(8, 257*ele['num_cnn_layers'], 8), ele['num_cnn_layers'])))
                list_conv1d_kernel_size.append(str(random.sample(range(2,8*ele['num_cnn_layers'],1), ele['num_cnn_layers'])))

        for i in range(12):
            ele['index'] = index
            index += 1
            ele['deep_layer_size'] = list_deep_layer_size[np.random.randint(0,150)]
            ele['conv1d_filters'] = list_conv1d_filters[np.random.randint(0,150)]
            ele['conv1d_kernel_size'] = list_conv1d_kernel_size[np.random.randint(0,150)]
            if ele['dilation']:
                ele['dilation_rate_size'] = list_dilation_rate_size[np.random.randint(0,150)]
            ele['epochs'] = 20000
            ele['patience'] = 400
            ele['input_shape'] = (400,1)
            #myexecute(f'echo "{json.dumps(ele,indent=4)}"')
            new_list_of_dicts.append(ele.copy())

    with open(list_of_dicts_dir, 'w') as f:
        json.dump(new_list_of_dicts, f)
    
    myexecute(f'echo "Created all search combinations in {time.time() - start_time} seconds"')
    myexecute(f'rsync -q -Pav {list_of_dicts_dir} {root_dir}hyperparameter_tuning/{model_type}/ ')
    list_of_dicts = new_list_of_dicts


def build_model(param_dict):

    num_cnn_layers = param_dict['num_cnn_layers']
        
    input_shape = param_dict['input_shape']

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    dilation = param_dict['dilation'] #take true or false as input
    batch_normalization = param_dict['batch_normalization'] #take true or false as input
    if dilation:
        dilation_rate_size = ast.literal_eval(param_dict['dilation_rate_size'])
    #conveting the strings to lists
    conv1d_filters = ast.literal_eval(param_dict['conv1d_filters'])
    conv1d_kernel_size = ast.literal_eval(param_dict['conv1d_kernel_size'])
    dense_units = ast.literal_eval(param_dict['deep_layer_size'])

    if dilation:
        for i in range(num_cnn_layers):
            x = layers.Conv1D(filters=conv1d_filters[i],
                    kernel_size=conv1d_kernel_size[i],
                    padding=param_dict['padding'],
                    dilation_rate=dilation_rate_size[i])(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

    else:
        for i in range(num_cnn_layers):
            x = layers.Conv1D(filters=conv1d_filters[i],
                              kernel_size=conv1d_kernel_size[i],
                              padding=param_dict['padding'],)(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

    #x = Dropout(param_dict['dropout_rate'])(x)
    x = layers.Flatten()(x)
    
    num_deep_layers = param_dict['num_deep_layers']
    for j in range(num_deep_layers):
        x = layers.Dense(dense_units[j],activation='relu')(x)
    
    final_outputs = layers.Dense(2, activation='relu')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=param_dict['initial_learning_rate'],
    decay_steps=10000,
    decay_rate=param_dict['decay_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
              loss='mse', metrics=['mse', 'mae'])
    return model

param_dict = list_of_dicts[tune_index]
input_shape = param_dict['input_shape']
batch_size = param_dict['batch_size']
epochs = 20000#param_dict['epochs']
patience = 200#param_dict['patience']
train_generator = create_dataset(X_train, Y_train, batch_size=batch_size, shuffle=True)
val_generator = create_dataset(X_val, Y_val, batch_size=batch_size, shuffle=False)

myexecute(f'echo "Starting the training for index {tune_index}"')
model = build_model(param_dict)
model.summary()

# Set up the ModelCheckpoint callback
checkpoint_filepath = best_model_name+'checkpoint.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Define early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop, checkpoint_callback], verbose=2)

model.save(best_model_name+'model.h5')

# Load the best model after training
# best_model = tf.keras.models.load_model(best_model_name+'checkpoint.h5', custom_objects={'custom_tanh_activation': custom_tanh_activation,'weighted_mse_loss': weighted_mse_loss, 'custom_sigmoid_activation': custom_sigmoid_activation,
#                                                                                         'weighted_mse_loss_wrapper' : get_weighted_mse_loss(default_hyperparameters['power'],default_hyperparameters['factor_mse'])})

best_model = model
# Test model on X_test
Y_pred = best_model.predict(X_test)
#get the loss on the test set
loss = best_model.evaluate(X_test, Y_test, verbose=0)
val_loss = best_model.evaluate(X_val, Y_val, verbose=0)
Y_pred = np.reshape(Y_pred, Y_test.shape)
# Y_pred_array = np.zeros_like(Y_test)
# Y_pred_array[:,0] = Y_pred[0].reshape(-1)
# Y_pred_array[:,1] = Y_pred[1].reshape(-1)
# Y_pred = Y_pred_array

accuracy = period_accuracy(Y_test, Y_pred)
median = median_percent_deviation(Y_test, Y_pred)
result_dict = {'index':tune_index,'accuracy':accuracy,'median':median,'loss':float(loss[0]),'val_loss':float(val_loss[0])}
#save the results
with open(f'{cur_log_dir}result_{tune_index}_{job_id}.json', 'w') as f:
    json.dump(result_dict, f)


myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The accuracy of the model is:{accuracy} \n\
          The median absolute error of the model is:{median} \n\
          The loss of the model on test data is :{loss} \n\
            The loss of the model on validation data is :{val_loss} \n\
          \n\n\n ############################################################################## \n\n\n \"')

#myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}checkpoint.h5 {root_dir}models/ ')
#myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}model.h5 {root_dir}models/ ')
myexecute(f'rsync -q -Pav {os.path.join(cur_log_dir, f"result_{tune_index}_{job_id}.json")} {root_dir}hyperparameter_tuning/{model_type}/ ')
if cur_dir != f'{root_dir}': 
    myexecute(f'rm -rf {cur_dir}')

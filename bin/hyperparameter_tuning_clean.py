from bin.hyperparameter_tuner_class import ModelTest,ActivationFunctions,LossFunctions,Metrics
import tensorflow as tf
import ast
import numpy as np
import os
import glob
import time
import json
import itertools
import random
import argparse


def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

def build_model(param_dict,**kwargs):
    num_deep_layers = param_dict['num_deep_layers']
    input_shape = param_dict['input_shape'] #input shape is a tuple like (800,1)
    min_label = param_dict['min_label']
    max_label = param_dict['max_label']
    activations = ActivationFunctions(min=min_label,max=max_label)
    losses = LossFunctions()
    metrics = Metrics()

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    dense_units = ast.literal_eval(param_dict['deep_layer_size'])
    for j in range(num_deep_layers):
        x = tf.keras.layers.Dense(dense_units[j],activation='relu')(x)
        
    x = tf.keras.layers.Dropout(rate=param_dict['dropout'])(x)
    x = tf.keras.layers.Flatten()(x)
    
    final_activation = param_dict['final_activation']
    if final_activation == 'sigmoid':
        final_outputs = tf.keras.layers.Dense(1, activation=activations.get_custom_sigmoid_activation)(x)
    elif final_activation == 'relu':
        final_outputs = tf.keras.layers.Dense(1, activation=activations.get_custom_relu_activation)(x)
    elif final_activation == 'tanh':
        final_outputs = tf.keras.layers.Dense(1, activation=activations.get_custom_tanh_activation)(x)
    elif final_activation == 'def_relu':
        final_outputs = tf.keras.layers.Dense(1, activation='relu')(x)
    else:
        raise ValueError('Invalid final activation function, use "none" if a simple relu activation is needed')
    
    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)
    power = param_dict['power']
    factor_mse = param_dict['factor_mse']
    loss_function = param_dict['loss_function']
    if loss_function == 'mse':
         loss = 'mse'
    else:
        loss = losses.get_weighted_mse_loss(power,factor_mse)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=param_dict['initial_learning_rate'],
    decay_steps=10000,
    decay_rate=param_dict['decay_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
            loss=loss, metrics=['mse', 
                                'mae',
                                metrics.get_period_accuracy(rat_max = 1.002,rat_min = 0.998),
                                metrics.get_median_percent_deviation(quartile = 50)])
    return model


def main(args):

    run = args.run
    log_dir = args.log_dir
    list_of_dicts = json.load(open(args.hyperparameters_loc))
    index = args.hyperparameters_index
    hyperparameters = list_of_dicts[index]
    best_model_name = os.path.join(cur_dir,log_dir,f'models/best_model_{index}_checkpoint.h5')
    cur_dir = args.cur_dir + str(int(index)) + '/'
     
    run = 'runBB'
    cur_dir = f'/tmp/Abhinav_DATA{index}/'
    #model_type = 'LSTM'
    input_shape = (400,1)
    batch_size = 1200
    epochs = 20000
    patience = 400
    best_model_name = f'{cur_dir}models/tuner_{index}_'
    root_dir = '/hercules/scratch/atya/BinaryML/'

    myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
            Comment: Binary simulations predictor: Predicting the frequency and drift. \nBest model is under:{best_model_name} \
            \n\n\n ############################################################################## \n\n\n \"')

    myexecute(f'mkdir -p {cur_dir}raw_data/{run}/')
    myexecute(f'mkdir -p {cur_dir}models/')

    #files to sync
    files1 = glob.glob(f'{root_dir}raw_data/{run}/*chunk.npy')
    # files1.extend(glob.glob(f'{root_dir}raw_data/{run}/*labels_{run}.npy'))
    for file in files1:
        myexecute(f'rsync -Pav -q {file} {cur_dir}raw_data/{run}/')

    # freq_axis = np.fft.rfftfreq(17280000, d=64e-6)
    # freq_res = freq_axis[1]-freq_axis[0]

    X_train = np.load(cur_dir + f'raw_data/{run}/train_data_chunk.npy').astype(np.float64)
    X_test = np.load(cur_dir + f'raw_data/{run}/test_data_chunk.npy').astype(np.float64)
    X_val = np.load(cur_dir + f'raw_data/{run}/val_data_chunk.npy').astype(np.float64)
    X_train = X_train/np.max(X_train,axis=1)[:,None]
    X_test = X_test/np.max(X_test,axis=1)[:,None]
    X_val = X_val/np.max(X_val,axis=1)[:,None]

    Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_chunk.npy').astype(np.float64)
    Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_chunk.npy').astype(np.float64)
    Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_chunk.npy').astype(np.float64)

    Y_train = 2*np.abs(Y_train[:,1])
    Y_test = 2*np.abs(Y_test[:,1])
    Y_val = 2*np.abs(Y_val[:,1])
    # Y_val = np.sort(Y_val,axis=1)

    # Y_train = Y_train/400
    # Y_test = Y_test/400
    # Y_val = Y_val/400

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

    # Y_train = (Y_train.reshape(Y_train.shape[0],Y_train.shape[1], 1))
    # Y_val = (Y_val.reshape(Y_val.shape[0],Y_val.shape[1], 1))
    #Y_test = (Y_test.reshape(Y_test.shape[0],Y_test.shape[1], 1))

    Y_train = (Y_train.reshape(Y_train.shape[0],1))
    Y_val = (Y_val.reshape(Y_val.shape[0],1))
    Y_test = (Y_test.reshape(Y_test.shape[0],1))

    myexecute(f'echo "Training data shape: {X_train.shape}"')
    myexecute(f'echo "Test data shape: {X_test.shape}"')
    myexecute(f'echo "Validation data shape: {X_val.shape}"')
    myexecute(f'echo "Training labels shape: {Y_train.shape}"')
    myexecute(f'echo "Test labels shape: {Y_test.shape}"')
    myexecute(f'echo "Validation labels shape: {Y_val.shape}"')
    print(Y_test[0:10])


    myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
        Comment: Starting the search for the best model\
        \n\n\n ############################################################################## \n\n\n \"')

    start_time = time.time()
    model_test = ModelTest(X_train,Y_train,X_test,Y_test,X_val,Y_val,cur_dir)
    best_model_trial, results_dict = model_test.train_and_evaluate_model(hyperparameters, build_model, num_trials = 3,verbose=1)
    print('\n\n####################################################################')
    print("Time taken: %s" % (time.time() - start_time))
    print('####################################################################\n\n')
    results_dict['time_taken'] = time.time() - start_time
    results_dict['trial'] = hyperparameters['trial']

    with open(f"{cur_dir+log_dir}results_dict_{int(results_dict['trial'])}.json", 'w') as f:
                json.dump(results_dict, f)

    best_model = best_model_trial            
    if best_model:
        metrics = Metrics()
        Y_pred = best_model.predict(X_test)

        accuracy = metrics.period_accuracy(Y_test, Y_pred)
        median = metrics.median_percent_deviation(Y_test, Y_pred)
        mean = metrics.mean_absolute_error(Y_test, Y_pred)

        myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
                Results on the test data are: \n \
                The accuracy of the model is:{0} \n\
                The median absolute error of the model is:{1} \n\
                The mean absolute error of the model is:{2} \n\
                \n\n\n ############################################################################## \n\n\n \"'.format(str(np.round(accuracy.numpy()*100,2)), median, mean))

        #myexecute(f'rsync -Pav {os.path.join(cur_dir, best_model_name)}model.h5 /hercules/scratch/atya/IsolatedML/models/ ')
        # Save the best model
        #best_model.save('/hercules/scratch/atya/IsolatedML/'+best_model_name+'model.h5')

    myexecute(f'rsync -Pav {os.path.join(cur_dir, log_dir)}* /hercules/scratch/atya/IsolatedML/{log_dir}/ ')

    if cur_dir != '/hercules/scratch/atya/IsolatedML/': 
        myexecute(f'rm -rf {cur_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hyperparameters_index', type=int, help="Index of the hyperparameters to be used")
    parser.add_argument('--run', type=str, default='runI')
    parser.add_argument('--cur_dir', type=str, default='/tmp/Abhinav_DATA')
    parser.add_argument('--log_dir', type=str, default='/hercules/scratch/atya/BinaryML/logs')
    parser.add_argument('--hyperparameters_loc', type=str, )

    args = parser.parse_args()
    main(args)



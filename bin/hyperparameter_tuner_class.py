import csv
import sys
import os
import glob
import ast
import json
import time
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

class ActivationFunctions():
    def __init__(self,min=min,max=max):
        self.min = min
        self.max = max

    def get_custom_tanh_activation(self,x):
        def custom_tanh_activation_wrapper(x):
            scaled_tanh = (tf.math.tanh(x) + 1) / 2
            range_scale = self.max / tf.reduce_max(scaled_tanh)  # maximum value is scaled to 0.029
            return range_scale * scaled_tanh + self.min # output is shifted by 0.001
        return custom_tanh_activation_wrapper(x)

    def get_custom_sigmoid_activation(self,x):
        def custom_sigmoid_activation_wrapper(x):
            return self.max / (1 + K.exp(-x)) + self.min
        return custom_sigmoid_activation_wrapper(x)
    
    def get_custom_relu_activation(self,x):
        def custom_relu_activation_wrapper(x):
            return K.minimum(K.maximum(self.min, x), self.max)
        return custom_relu_activation_wrapper(x)
    
class LossFunctions():
    def __init__(self) -> None:
        pass

    # Custom loss function with requiring higher precision for lower periods
    def weighted_mse_loss(self,y_true, y_pred, power,factor_mse=1.0):
        # Calculate squared error
        y_true = y_true*factor_mse
        y_pred = y_pred*factor_mse
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        squared_error = mse(y_true, y_pred)
        
        # Calculate the weighting factor (you can adjust the scaling factor as needed)
        weighting_factor = 1.0 / (y_true + 1e-8)**power  # Adding a small constant to avoid division by zero
        
        # Apply the weighting factor to the squared error
        weighted_squared_error = weighting_factor * squared_error
        
        # Calculate the mean of the weighted squared error
        weighted_mse = tf.reduce_mean(weighted_squared_error)
        
        return weighted_mse

    def get_weighted_mse_loss(self,power,factor_mse):
        def weighted_mse_loss_wrapper(y_true, y_pred):
            return self.weighted_mse_loss(y_true, y_pred, power,factor_mse)
        return weighted_mse_loss_wrapper
    
class Metrics():
    def __init__(self) -> None:
        pass

    def get_period_accuracy(self,rat_max = 1.002,rat_min = 0.998):
        def period_accuracy_wrapper(Y_test, Y_pred):
            return self.period_accuracy(Y_test,Y_pred,rat_max,rat_min)
        return period_accuracy_wrapper

    def get_median_percent_deviation(self,quartile=50):
        def median_percent_deviation_wrapper(Y_test,Y_pred):
            return self.median_percent_deviation(Y_test,Y_pred,quartile)
        return median_percent_deviation_wrapper
    
    def get_mean_squared_error(self):
        def mean_squared_error_wrapper(Y_test,Y_pred):
            return self.mean_squared_error(Y_test,Y_pred)
        return mean_squared_error_wrapper
    
    def get_mean_absolute_error(self):
        def mean_absolute_error_wrapper(Y_test,Y_pred):
            return self.mean_absolute_error(Y_test,Y_pred)
        return mean_absolute_error_wrapper
    
    def get_r2_score(self):
        def r2_score_wrapper(Y_test,Y_pred):
            return self.r2_score(Y_test,Y_pred)
        return r2_score_wrapper

    def period_accuracy(self, Y_test, Y_pred, rat_max = 1.002,rat_min = 0.998):
        """
        Computes the percentage of predicted periods that are within 10% of the true periods.
        """
        rat = (Y_test+ 1e-10)/(Y_pred+ 1e-10)
        #now if the ratio is less than 1.002 and greater than 0.998, then return 1
        rat_bool = tf.logical_and(tf.less(rat, rat_max), tf.greater(rat, rat_min))
        return tf.reduce_mean(tf.cast(rat_bool, tf.float32)) * 100

    def median_percent_deviation(self,Y_test,Y_pred,quartile=50):
        """
        Computes the median percent deviation from the predicted period.
        """
        percent_deviation = tf.abs((Y_test - Y_pred) / (Y_test + 1e-10)) * 100
        median_deviation = tfp.stats.percentile(percent_deviation, q=quartile)
        return median_deviation
    
    def mean_squared_error(self,Y_test,Y_pred):
        return mean_squared_error(Y_test, Y_pred)
    
    def mean_absolute_error(self,Y_test,Y_pred):
        return mean_absolute_error(Y_test, Y_pred)
    
    def r2_score(self,Y_test,Y_pred):
        return r2_score(Y_test, Y_pred)

class ModelTest():
    def __init__(self,X_train,Y_train,X_test,Y_test,X_val,Y_val,cur_dir,activation_functions,loss_functions,metrics) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.cur_dir = cur_dir
        self.activation_functions = activation_functions
        self.loss_functions = loss_functions
        self.metrics = metrics
        pass

    def create_dataset(self,X, y, batch_size, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset
    
    def train_generator(self,batch_size,shuffle):
        return self.create_dataset(self.X_train, self.Y_train, batch_size=batch_size, shuffle=shuffle)
    
    def val_generator(self,batch_size,shuffle):
        return self.create_dataset(self.X_val, self.Y_val, batch_size=batch_size, shuffle=shuffle)
    
    def test_generator(self,batch_size,shuffle):
        return self.create_dataset(self.X_test, self.Y_test, batch_size=batch_size, shuffle=shuffle)
    
    def train_and_evaluate_model(self,hyperparameters, build_fn, max_seconds=13800,num_trials = 3,verbose=1,**kwargs):
        
        trial_index = hyperparameters['trial']
        patience = hyperparameters['patience']
        epochs = hyperparameters['epoches']
        val_generator = self.val_generator(batch_size=hyperparameters['batch_size'],shuffle=False)
        train_generator = self.train_generator(batch_size=hyperparameters['batch_size'],shuffle=True)
    
        checkpoint_filepath = os.path.join(self.cur_dir,f'best_model_checkpoint_{trial_index}.h5')
        # check if the training needs to be resumed or started from scratch
        # check if the checkpoint exists already else start training from scratch
        if os.path.exists(checkpoint_filepath):
            # check if results dict exists else raise error
            if os.path.exists(os.path.join(self.cur_dir,f'results_dict_{hyperparameters["trial"]}.json')):
                results_dict = json.load(open(os.path.join(self.cur_dir,f'results_dict_{hyperparameters["trial"]}.json'),'r'))
                resume_training = True
            else:
                raise ValueError('The checkpoint exists but the results_dict does not exist. Please check the code')
        else:
            # initialize all the metrics
            results_dict = {'accuracy':0,
                            'median': 0,
                                'mse': 0,
                                'mae': 0,
                                'val_loss': float('inf'),
                                'n_trial':0,
                                'trial_index':trial_index,
                                }
            resume_training = False

        
        best_val_loss_trial = results_dict['val_loss']
        best_model_trial = None
        best_accuracy_trial, best_median_trial, best_mse_trial, best_mae_trial = results_dict['accuracy'], results_dict['median'], results_dict['mse'], results_dict['mae']
 
        n_trial = results_dict['n_trial']
        first_trial_in_this_run = True
        for p in range(n_trial,num_trials):
                    
            if verbose == 1:
                myexecute(f'echo "\n\n\n\n##############################################################################\n\n\n\
                Comment: Model trial number: {trial_index} for the {p} time \n\
                \n##############################################################################\n\n\n"')

            model = build_fn(hyperparameters,**kwargs)
            if resume_training:
                model.load_weights(checkpoint_filepath)  
                if verbose == 1:
                    myexecute(f'echo "\n\n\n\n##############################################################################\n\n\n\
                    Comment: Resuming training from the previous best weights: {trial_index} for the {p} time \n\
                    \n##############################################################################\n\n\n"')

            if verbose == 1:
                if first_trial_in_this_run:
                    model.summary()
                    first_trial_in_this_run = False


            # Define early stopping callback
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)

            # Set up the ModelCheckpoint callback

            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )

            # Set up the TimeLimitCallback callback
            class TimeLimitCallback(tf.keras.callbacks.Callback):
                def __init__(self, max_seconds):
                    self.max_seconds = max_seconds
                    self.start_time = None

                def on_train_begin(self, logs=None):
                    self.start_time = time.time()

                def on_epoch_end(self, epoch, logs=None):
                    if time.time() - self.start_time > self.max_seconds:
                        self.model.stop_training = True
                        print("\nReached time limit. Stopping training...")
                
                def on_train_end(self, logs=None):
                    if time.time() - self.start_time > self.max_seconds:
                        self.model.stop_training = True
                        myexecute('echo "Stopped training due to time limit"')
            
            time_limit_callback = TimeLimitCallback(max_seconds=max_seconds)
                    

            history = model.fit(train_generator, epochs=epochs, validation_data=val_generator,
                                 callbacks=[early_stop,checkpoint_callback,time_limit_callback],verbose=2)

            # Evaluate the model
            val_loss_trial, mse_trial, mae_trial, accuracy_trial, median_trial = model.evaluate(val_generator, verbose=2)

            if val_loss_trial < best_val_loss_trial:
                best_val_loss_trial = val_loss_trial
                best_model_trial = model
                best_accuracy_trial, best_median_trial, best_mse_trial, best_mae_trial = accuracy_trial, median_trial, mse_trial, mae_trial
            
            # #Clear all model variable from previous session
            tf.keras.backend.clear_session()

        results_dict = {'accuracy':best_accuracy_trial,
                        'median': best_median_trial,
                            'mse': best_mse_trial,
                            'mae': best_mae_trial,
                            'val_loss': best_val_loss_trial,
                            'n_trial':p,
                            'trial_index':trial_index,
                            }
                            
        model = best_model_trial

        return best_model_trial, results_dict

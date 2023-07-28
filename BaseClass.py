# contains all the base functions for other programs to be used
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
from tensorflow.keras.layers import MultiHeadAttention, Attention
import json
import time
import pandas as pd
import argparse

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

class Models():
    def __init__(self,param_dict):
        self.param_dict = param_dict
    def cnn(self):
        num_cnn_layers = self.param_dict['num_cnn_layers']
            
        input_shape = self.param_dict['input_shape']

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        dilation = self.param_dict['dilation'] #take true or false as input
        batch_normalization = self.param_dict['batch_normalization'] #take true or false as input
        if dilation:
            dilation_rate_size = ast.literal_eval(self.param_dict['dilation_rate_size'])
        #conveting the strings to lists
        conv1d_filters = ast.literal_eval(self.param_dict['conv1d_filters'])
        conv1d_kernel_size = ast.literal_eval(self.param_dict['conv1d_kernel_size'])
        dense_units = ast.literal_eval(self.param_dict['deep_layer_size'])

        if dilation:
            for i in range(num_cnn_layers):
                x = layers.Conv1D(filters=conv1d_filters[i],
                        kernel_size=conv1d_kernel_size[i],
                        padding=self.param_dict['padding'],
                        dilation_rate=dilation_rate_size[i])(x)
                if batch_normalization:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

        else:
            for i in range(num_cnn_layers):
                x = layers.Conv1D(filters=conv1d_filters[i],
                                kernel_size=conv1d_kernel_size[i],
                                padding=self.param_dict['padding'],)(x)
                if batch_normalization:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

        #x = Dropout(self.param_dict['dropout_rate'])(x)
        x = layers.Flatten()(x)
        
        num_deep_layers = self.param_dict['num_deep_layers']
        for j in range(num_deep_layers):
            x = layers.Dense(dense_units[j],activation='relu')(x)
        
        final_outputs = layers.Dense(1, activation='relu')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=final_outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.param_dict['initial_learning_rate'],
        decay_steps=10000,
        decay_rate=self.param_dict['decay_rate'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                loss='mse', metrics=['mse', 'mae'])
        return model
    
    def attention(self):
        num_cnn_layers = self.param_dict['num_cnn_layers']
            
        input_shape = self.param_dict['input_shape']

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs

        batch_normalization = self.param_dict['batch_normalization'] #take true or false as input
        #conveting the strings to lists
        conv1d_filters = ast.literal_eval(self.param_dict['conv1d_filters'])
        conv1d_kernel_size = ast.literal_eval(self.param_dict['conv1d_kernel_size'])
        dense_units = ast.literal_eval(self.param_dict['deep_layer_size'])

        for i in range(num_cnn_layers):
                x = layers.Conv1D(filters=conv1d_filters[i],
                                kernel_size=conv1d_kernel_size[i],
                                padding='same',)(x)
                if batch_normalization:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

        attention_type = self.param_dict['attention_type']
        num_attention_layers = self.param_dict['num_attention_layers']
        if attention_type == 'multi_head':
            num_attention_heads = self.param_dict['num_attention_heads']
            num_key_dims = self.param_dict['num_key_dims']
            for i in range(num_attention_layers):
                x = MultiHeadAttention(num_heads=num_attention_heads, key_dim=num_key_dims)(x,x)
        elif attention_type == 'simple':
            for i in range(num_attention_layers):
                x = layers.Attention()([x,x])
        #x = Dropout(self.param_dict['dropout_rate'])(x)
        x = layers.Flatten()(x)
        
        num_deep_layers = self.param_dict['num_deep_layers']
        for j in range(num_deep_layers):
            x = layers.Dense(dense_units[j],activation='relu')(x)
        
        final_outputs = layers.Dense(1, activation='relu')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=final_outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.param_dict['initial_learning_rate'],
        decay_steps=10000,
        decay_rate=self.param_dict['decay_rate'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                loss='mse', metrics=['mse', 'mae'])
        return model
    
class CallBacks():
    def __init__(self):
        pass

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

class TrainUtils():
    def __init__(self):
        pass

    def resume_training(self,previous_job_id,root_dir,cur_dir):
        file_name_checkpoint = None
        best_model_name_pattern = f'{root_dir}models/*{previous_job_id}*'
        best_models = glob.glob(best_model_name_pattern)
        if len(best_models) == 0:
            raise ValueError('No previous model found')
        elif len(best_models) > 1:
            raise ValueError('Need a unique job id to resume training')
        else:
            file_name_checkpoint_root = best_models[0]
            myexecute(f'rsync -Pav -q {file_name_checkpoint_root} {cur_dir}models/')
            file_name_checkpoint = file_name_checkpoint_root.replace(root_dir,cur_dir)
        return file_name_checkpoint
    
    def find_indices(self,small_list, big_list):
        indices = []

        for item in small_list:
            if item in big_list:
                index = np.where(big_list == item)[0]
                indices.append(index[0])

        return indices
    
def Metrics():
    def __init__(self):
        pass

    def median_percent_deviation(self,Y_test, Y_pred):
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

        return [med.numpy() for med in median_deviations]

    def period_accuracy(self,Y_test, Y_pred):
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
        return [acc.numpy() for acc in accuracies]

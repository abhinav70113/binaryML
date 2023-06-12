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

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)


job_id = sys.argv[1]
model_type = sys.argv[2]
run = 'runBB'
cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
#model_type = 'LSTM'
input_shape = (400,1)
batch_size = 1200
epochs = 20000
patience = 400
best_model_name = f'{cur_dir}models/z_predict_{model_type}_{job_id}_'
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


# z_val = Y_val[:,1].copy()
# Y_val[:,0] = Y_val[:,0] - z_val
# Y_val[:,1] = Y_val[:,0] + z_val
# Y_val = np.sort(Y_val,axis=1)

# z_test = Y_test[:,1].copy()
# Y_test[:,0] = Y_test[:,0] - z_test
# Y_test[:,1] = Y_test[:,0] + z_test
# Y_test = np.sort(Y_test,axis=1)

# z_train = Y_train[:,1].copy()
# Y_train[:,0] = Y_train[:,0] - z_train
# Y_train[:,1] = Y_train[:,0] + z_train
# Y_train = np.sort(Y_train,axis=1)

# Y_train = 2*np.abs(Y_train[:,1])
# Y_test = 2*np.abs(Y_test[:,1])
# Y_val = 2*np.abs(Y_val[:,1])

Y_train[:,1] = 2*np.abs(Y_train[:,1])
Y_test[:,1] = 2*np.abs(Y_test[:,1])
Y_val[:,1] = 2*np.abs(Y_val[:,1])

# Y_train = Y_train/400
# Y_test = Y_test/400
# Y_val = Y_val/400

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

Y_train = (Y_train.reshape(Y_train.shape[0],Y_train.shape[1], 1))
Y_val = (Y_val.reshape(Y_val.shape[0],Y_val.shape[1], 1))
Y_test = (Y_test.reshape(Y_test.shape[0],Y_test.shape[1], 1))

# Y_train = (Y_train.reshape(Y_train.shape[0],1))
# Y_val = (Y_val.reshape(Y_val.shape[0],1))
# Y_test = (Y_test.reshape(Y_test.shape[0],1))

myexecute(f'echo "Training data shape: {X_train.shape}"')
myexecute(f'echo "Test data shape: {X_test.shape}"')
myexecute(f'echo "Validation data shape: {X_val.shape}"')
myexecute(f'echo "Training labels shape: {Y_train.shape}"')
myexecute(f'echo "Test labels shape: {Y_test.shape}"')
myexecute(f'echo "Validation labels shape: {Y_val.shape}"')
print(Y_test[0:10])


# Create data generators
def create_dataset(X, y, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_generator = create_dataset(X_train, Y_train, batch_size=batch_size, shuffle=True)
val_generator = create_dataset(X_val, Y_val, batch_size=batch_size, shuffle=False)

def custom_tanh_activation(x):
    scaled_tanh = (tf.math.tanh(x) + 1) / 2
    range_scale = 0.029 / tf.reduce_max(scaled_tanh)  # maximum value is scaled to 0.029
    return range_scale * scaled_tanh + 0.001  # output is shifted by 0.001

def custom_sigmoid_activation(x):
    return 30 / (1 + K.exp(-x)) + 0.01

def weighted_mse_loss(y_true, y_pred):
    # Calculate squared error
    y_true = y_true
    y_pred = y_pred
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    squared_error = mse(y_true, y_pred)
    
    # Calculate the weighting factor (you can adjust the scaling factor as needed)
    weighting_factor = 1.0 / (y_true + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Apply the weighting factor to the squared error
    weighted_squared_error = weighting_factor * squared_error
    
    # Calculate the mean of the weighted squared error
    weighted_mse = tf.reduce_mean(weighted_squared_error)
    
    return weighted_mse

def custom_loss(Y_true, Y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mse_abs = 0
    #Y_test = tf.cast(Y_test, tf.float32)
    #Y_pred = tf.cast(Y_pred, tf.float32)
    mse_abs = tf.reduce_mean(mse(Y_true[:,0], Y_pred[:,0]))
    mse_abs += (tf.reduce_mean(mse(Y_true[:,1], Y_pred[:,1]))*16) #16 is the factor since Y_true only varies in 1/4 th the range of 0th index
    return mse_abs


def get_weighted_mse_loss(power,factor_mse):
    def weighted_mse_loss_wrapper(y_true, y_pred):
        return weighted_mse_loss(y_true, y_pred, power,factor_mse)
    return weighted_mse_loss_wrapper

class SortingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SortingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.sort(inputs, direction='ASCENDING', axis=-1)

# # Define custom metric function
# def period_accuracy(Y_test, Y_pred):
#     """
#     Computes the percentage of predicted periods that are within 10% of the true periods.
#     """
#     rat = (Y_test+ 1e-10)/(Y_pred+ 1e-10)
#     #now if the ratio is less than 1.1 and greater than 0.9, then return 1
#     rat_bool = tf.logical_and(tf.less(rat, 1.005), tf.greater(rat, 0.995))
#     return tf.reduce_mean(tf.cast(rat_bool, tf.float32)) * 100

# def median_percent_deviation(Y_test, Y_pred):
#     """
#     Computes the median percent deviation from the predicted period.
#     """
#     percent_deviation = tf.abs((Y_test - Y_pred) / (Y_test + 1e-10)) * 100
#     median_deviation = tfp.stats.percentile(percent_deviation, q=50)
#     return median_deviation

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
    return [acc.numpy() for acc in accuracies]


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

    return [med.numpy() for med in median_deviations]

def custom_tanh_activation(x):
    scaled_tanh = (tf.math.tanh(x) + 1) / 2
    range_scale = 30 / tf.reduce_max(scaled_tanh)  # maximum value is scaled to 0.029
    return range_scale * scaled_tanh + 0.00001  # output is shifted by 0.001

def custom_sigmoid_activation(x):
    return 1 / (1 + K.exp(-x)) + 0.00001

def custom_relu_activation(x):
    return K.minimum(K.maximum(0.00001, x), 30)

# Define a helper function for a Residual block
def ResidualBlock(x, filters, kernel_size):
    skip = x
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(skip) # to match dimensions
    return tf.keras.layers.Add()([x, skip]) # Skip connection
def LSTM_model(input_shape = (400,2)):

    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = ResidualBlock(inputs, filters=64, kernel_size=5)
    x = ResidualBlock(x, filters=256, kernel_size=7)

    # Convert the output shape of ResidualBlock to 3D for LSTM
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)

    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(192, return_sequences=False)(x)
    outputs = tf.keras.layers.Dense(2, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def attention_model(input_shape = (400,2)):
    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)

    # Apply Conv1D layers
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',activation='relu')(x)
    # Apply Multihead Attention
    x = MultiHeadAttention(num_heads=2, key_dim=2)(x,x)
    #x = MultiHeadAttention(num_heads=4, key_dim=2)(x,x)
    #x = MultiHeadAttention(num_heads=8, key_dim=2)(x,x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(192, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def dense_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer

    outputs = tf.keras.layers.Dense(2, activation='relu')(x)  # change activation function to 'sigmoid'
    outputs = SortingLayer()(outputs)  # Sort the output values
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def cnn_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    #x = ResidualBlock(inputs, filters=192, kernel_size=5)
    #x = ResidualBlock(x, filters=240, kernel_size=7)
    #x = tf.keras.layers.Flatten()(x)

    # x = layers.Conv1D(64, 3)(inputs)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = layers.MaxPooling1D(2)(x)
    
    # x = layers.Conv1D(128, 5)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = layers.MaxPooling1D(2)(x)
    
    # x = layers.Conv1D(256, 7)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = layers.Conv1D(512, 4)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = layers.Conv1D(1024, 3)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = layers.Conv1D(2048, 3)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = layers.Conv1D(2048, 3)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    # x = layers.Conv1D(2048, 3)(x)
    # #x = layers.MaxPooling1D(2)(x)
    # #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # # Apply Conv1D layers
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same',dilation_rate = 2,activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same',dilation_rate = 4,activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same',dilation_rate = 8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 16, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 32, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def resnet_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = ResidualBlock(inputs, filters=64, kernel_size=3)
    x = ResidualBlock(x, filters=256, kernel_size=3)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer

    outputs = tf.keras.layers.Dense(2, activation='relu')(x)
    #x = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def current_model(model_name = "deep",input_shape = (400,2)):
    if model_name == 'attention':
        return attention_model(input_shape=input_shape)
    elif model_name == 'deep':
        return dense_model(input_shape=input_shape)
    elif model_name == 'resnet':
        return resnet_model(input_shape=input_shape)
    elif model_name == 'cnn':
        return cnn_model(input_shape=input_shape)
    elif model_name == 'LSTM':
        return LSTM_model(input_shape=input_shape)
    else:
        raise ValueError('Model name not found')

model = current_model(model_name = model_type,input_shape=input_shape)
######################################

#Learning rate exponentia

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model with mean squared error loss
model.compile(optimizer=optimizer, loss=custom_loss, metrics=[custom_loss,'mse', 'mae'])

# def custom_mae(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_true - tf.argmax(y_pred, axis=-1)))

# def custom_mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - tf.argmax(y_pred, axis=-1)))

# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy', 
#               metrics=[custom_mae, custom_mse])
# Compile the model with mean squared error loss
# model.compile(optimizer=optimizer, 
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])


# Print model summary
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
#best_model = tf.keras.models.load_model(best_model_name+'checkpoint.h5', custom_objects={'custom_tanh_activation': custom_tanh_activation,'weighted_mse_loss': weighted_mse_loss, 'custom_sigmoid_activation': custom_sigmoid_activation,
#                                                                                         'weighted_mse_loss_wrapper' : get_weighted_mse_loss(default_hyperparameters['power'],default_hyperparameters['factor_mse'])})

best_model = model
# Test model on X_test
Y_pred = best_model.predict(X_test)
Y_pred = np.reshape(Y_pred, Y_test.shape)
# Y_pred_array = np.zeros_like(Y_test)
# Y_pred_array[:,0] = Y_pred[0].reshape(-1)
# Y_pred_array[:,1] = Y_pred[1].reshape(-1)
# Y_pred = Y_pred_array

accuracy = period_accuracy(Y_test, Y_pred)
median = median_percent_deviation(Y_test, Y_pred)


myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The accuracy of the model is:{accuracy} \n\
          The median absolute error of the model is:{median} \n\
          \n\n\n ############################################################################## \n\n\n \"')

myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}checkpoint.h5 {root_dir}models/ ')
myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}model.h5 {root_dir}models/ ')
if cur_dir != '{root_dir}': 
    myexecute(f'rm -rf {cur_dir}')

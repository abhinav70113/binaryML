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
from sklearn.metrics import accuracy_score

def period_accuracy(y_true, y_pred):
    # Ensure the predictions are the same shape as the true values
    assert y_true.shape == y_pred.shape
    return accuracy_score(y_true, y_pred)

def myexecute(cmd):
    print("'%s'"%cmd)
    os.system("echo '%s'"%cmd)
    os.system(cmd)


job_id = sys.argv[1]
segment_index = int(sys.argv[2])
# period_res = 1/0.029802322387695312
# period_min = 0.001
# period_max = 0.03
# arg2 = sys.argv[2]
# cur_dir = arg1
# best_model_name = os.path.join(cur_dir, arg2)
run = 'runJ'
segment = 1
cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
best_model_name = f'{cur_dir}models/trial_weak_learner_{run}_seg{segment}_{segment_index}_{job_id}_'
model_type = 'deep'
input_shape = (400,1)
#cur_dir = '/hercules/scratch/atya/IsolatedML/'

myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Running classification instead of regression and running now on runJ datset on fft of unsegmented datset but there is no frequency channel given alongside but the correct output is now between 1 and 400 and lies anywhere and not in the middle Best model is under:{0} \
           \n\n\n ############################################################################## \n\n\n \"'.format(best_model_name+'checkpoint.h5'))

myexecute(f'mkdir -p {cur_dir}raw_data/{run}/')
myexecute(f'mkdir -p {cur_dir}models/')
print(best_model_name, cur_dir)

#files to sync
files1 = glob.glob(f'/hercules/scratch/atya/IsolatedML/raw_data/{run}/*{run}_seg{segment}_{segment_index}_fftdat.npy')
files1.extend(glob.glob(f'/hercules/scratch/atya/IsolatedML/raw_data/{run}/*labels_{run}.npy'))
for file in files1:
    myexecute(f'rsync -Pav {file} {cur_dir}raw_data/{run}/')
    
seg_length = int(4194304/segment)
nyquist_rate = 7812.5
freq_axis = np.fft.rfftfreq(seg_length, d=1/(2*nyquist_rate))

if run == 'runIJKL':
    X_train = np.array(np.memmap(cur_dir + f'raw_data/{run}/train_data_{run}_seg{segment}_{segment_index}_fftdat.npy',dtype=np.float32,mode='r',shape=(4800,262145)))
    X_test = np.array(np.memmap(cur_dir + f'raw_data/{run}/test_data_{run}_seg{segment}_{segment_index}_fftdat.npy',dtype=np.float32,mode='r',shape=(1600,262145)))
    X_val = np.array(np.memmap(cur_dir + f'raw_data/{run}/val_data_{run}_seg{segment}_{segment_index}_fftdat.npy',dtype=np.float32,mode='r',shape=(1600,262145)))

else:
    X_train = np.load(cur_dir + f'raw_data/{run}/train_data_{run}_seg{segment}_{segment_index}_fftdat.npy').astype(np.float64)
    X_test = np.load(cur_dir + f'raw_data/{run}/test_data_{run}_seg{segment}_{segment_index}_fftdat.npy').astype(np.float64)
    X_val = np.load(cur_dir + f'raw_data/{run}/val_data_{run}_seg{segment}_{segment_index}_fftdat.npy').astype(np.float64)

Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_{run}.npy').astype(np.float64)
Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_{run}.npy').astype(np.float64)
Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_{run}.npy').astype(np.float64)

Y_train_freq = 1/Y_train
Y_test_freq = 1/Y_test
Y_val_freq = 1/Y_val

# Y_train = 1/Y_train
# Y_test = 1/Y_test
# Y_val = 1/Y_val

Y_train_arg = np.zeros_like(Y_train_freq)
for i in range(Y_train_freq.shape[0]):
    Y_train_arg[i] = np.argmin(np.abs(freq_axis - Y_train_freq[i]))

Y_test_arg = np.zeros_like(Y_test_freq)
for i in range(Y_test_freq.shape[0]):
    Y_test_arg[i] = np.argmin(np.abs(freq_axis - Y_test_freq[i]))

Y_val_arg = np.zeros_like(Y_val_freq)
for i in range(Y_val_freq.shape[0]):
    Y_val_arg[i] = np.argmin(np.abs(freq_axis - Y_val_freq[i]))
#Y_train_arg = np.argmin(np.abs(freq_axis - Y_train_freq[:,0,None]), axis=1)

Y_train_arg = Y_train_arg.astype(int)
Y_test_arg = Y_test_arg.astype(int)
Y_val_arg = Y_val_arg.astype(int)

fft_length = X_train.shape[1]
#convert fft length to a power of 2
fft_length = 2**int(np.log2(fft_length))

X_train = X_train[:,:fft_length]
X_test = X_test[:,:fft_length]
X_val = X_val[:,:fft_length]

fft_bins = 800
freq_res = freq_axis[1] - freq_axis[0]
freq_axis_chunks = fft_length//fft_bins 

Y_train = np.zeros_like(Y_train_arg)
Y_test = np.zeros_like(Y_test_arg)
Y_val = np.zeros_like(Y_val_arg)

X_train_freq = np.zeros((X_train.shape[0],fft_bins,2))
for i in range(X_train.shape[0]):
    freq = freq_axis[Y_train_arg[i]]
    for j in range(freq_axis_chunks):
        start = j * (fft_bins)
        end = start + fft_bins
        #print(start,end)
        freq_chunk = freq_axis[start:end]
        if (freq < freq_chunk[-1]) and (freq > freq_chunk[0]):
            X_train_freq[i,:,0] = X_train[i,start:end]
            #X_train_freq[i,:,1] = (1/freq_chunk)*1000
            X_train_freq[i,:,1] = np.arange(0,400,1).astype(np.float64)
            Y_train[i] = np.argmin(np.abs(freq_chunk - freq))
            break

X_test_freq = np.zeros((X_test.shape[0],fft_bins,2))
for i in range(X_test.shape[0]):
    freq = freq_axis[Y_test_arg[i]]
    for j in range(freq_axis_chunks):
        start = j * (fft_bins)
        end = start + fft_bins
        freq_chunk = freq_axis[start:end]
        if (freq < freq_chunk[-1]) and (freq > freq_chunk[0]):
            X_test_freq[i,:,0] = X_test[i,start:end]
            #X_test_freq[i,:,1] = (1/freq_chunk)*1000
            X_test_freq[i,:,1] = np.arange(0,400,1).astype(np.float64)
            Y_test[i] = np.argmin(np.abs(freq_chunk - freq))
            break

X_val_freq = np.zeros((X_val.shape[0],fft_bins,2))
for i in range(X_val.shape[0]):
    freq = freq_axis[Y_val_arg[i]]
    for j in range(freq_axis_chunks):
        start = j * (fft_bins)
        end = start + fft_bins
        freq_chunk = freq_axis[start:end]
        if (freq < freq_chunk[-1]) and (freq > freq_chunk[0]):
            X_val_freq[i,:,0] = X_val[i,start:end]
            #X_val_freq[i,:,1] = (1/freq_chunk)*1000
            X_val_freq[i,:,1] = np.arange(0,400,1).astype(np.float64)
            Y_val[i] = np.argmin(np.abs(freq_chunk - freq))
            break

X_train = X_train_freq[:,:,0] # take only the first channel
X_test = X_test_freq[:,:,0]
X_val = X_val_freq[:,:,0]

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1],X_test.shape[2], 1) 

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

Y_train = (Y_train.reshape(Y_train.shape[0], 1))
Y_test = (Y_test.reshape(Y_test.shape[0], 1))
Y_val = (Y_val.reshape(Y_val.shape[0], 1))

print('Training data shape: ', X_train.shape)
print('test labels',Y_test.shape)
print('training data sample for first row is: ', np.min(X_train[0,:40]))
#print('training data sample for second row is: ', np.min(X_train[0,:40,1],axis=1))
print('Minimum Spin Period in training data is: ', np.min(Y_train))
print('Minimum Spin Period in test data is: ', np.min(Y_test))
print('Minimum Spin Period in validation data is: ', np.min(Y_val))

   
# Create data generators
def create_dataset(X, y, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_generator = create_dataset(X_train, Y_train, batch_size=1200, shuffle=True)
val_generator = create_dataset(X_val, Y_val, batch_size=1200, shuffle=False)

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

def get_weighted_mse_loss(power,factor_mse):
    def weighted_mse_loss_wrapper(y_true, y_pred):
        return weighted_mse_loss(y_true, y_pred, power,factor_mse)
    return weighted_mse_loss_wrapper

# # Define custom metric function
# def period_accuracy(Y_test, Y_pred):
#     """
#     Computes the percentage of predicted periods that are within 10% of the true periods.
#     """
#     rat = (Y_test+ 1e-10)/(Y_pred+ 1e-10)
#     #now if the ratio is less than 1.1 and greater than 0.9, then return 1
#     rat_bool = tf.logical_and(tf.less(rat, 1.005), tf.greater(rat, 0.995))
#     return tf.reduce_mean(tf.cast(rat_bool, tf.float32)) * 100

def median_percent_deviation(Y_test, Y_pred):
    """
    Computes the median percent deviation from the predicted period.
    """
    percent_deviation = tf.abs((Y_test - Y_pred) / (Y_test + 1e-10)) * 100
    median_deviation = tfp.stats.percentile(percent_deviation, q=50)
    return median_deviation

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
    # The final Dense layer should have as many neurons as classes,
    # and use softmax activation for a multi-class classification problem
    x = layers.Flatten()(x)
    outputs = layers.Dense(400, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def attention_model(input_shape = (400,2)):
    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)

    # Apply Conv1D layers
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same', activation='relu')(x)

    # Apply Multihead Attention
    x = MultiHeadAttention(num_heads=2, key_dim=2)(x, x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(192, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(400, activation='softmax')(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def dense_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer

    x = layers.Flatten()(x)
    outputs = layers.Dense(400, activation='softmax')(x)

    #x = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def cnn_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    #x = ResidualBlock(inputs, filters=192, kernel_size=5)
    #x = ResidualBlock(x, filters=240, kernel_size=7)
    #x = tf.keras.layers.Flatten()(x)

    x = layers.Conv1D(64, 3, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, activation='relu')(x)
    #x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(512, 3, activation='relu')(x)
    #x = layers.MaxPooling1D(2)(x)

    x = layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer
    #x = layers.flatten()(x)
    outputs = layers.Dense(400, activation='softmax')(x)

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

    #x = layers.latten()(x)
    outputs = layers.Dense(400, activation='softmax')(x)

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
model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

# def custom_mae(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_true - tf.argmax(y_pred, axis=-1)))

# def custom_mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - tf.argmax(y_pred, axis=-1)))

# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy', 
#               metrics=[custom_mae, custom_mse])
#Compile the model with mean squared error loss
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# Print model summary
model.summary()

# Set up the ModelCheckpoint callback
checkpoint_filepath = best_model_name+'checkpoint.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Define early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=300, verbose=1, mode='max', restore_best_weights=True)

class MaxAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        accuracy = logs.get('val_accuracy')
        if accuracy is not None and accuracy >= 1.0:
            print("\nReached 100% accuracy, stopping training.")
            self.model.stop_training = True


max_accuracy = MaxAccuracy()
# Train the model
model.fit(train_generator, epochs=20000, validation_data=val_generator, callbacks=[early_stop, checkpoint_callback,max_accuracy], verbose=2)

model.save(best_model_name+'model.h5')

# Load the best model after training
best_model = tf.keras.models.load_model(best_model_name+'checkpoint.h5')

best_model = model
# Test model on X_test
Y_pred = (best_model.predict(X_test)).astype(int)
Y_pred = np.argmax(Y_pred, axis=-1)
Y_test = Y_test.reshape(Y_test.shape[0],)
myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The shape of predicted model is :{Y_pred.shape} \n\
          The shape of test model is :{Y_test.shape} \
          \n\n\n ########################################################')


accuracy = period_accuracy(Y_test, Y_pred)
# median = np.median(np.abs(Y_test-Y_pred)*100/Y_test)
# mean = np.mean(np.abs(Y_test-Y_pred)*100/Y_test)
# plt.scatter(Y_test,np.abs(Y_pred-Y_test)*100/Y_test)

myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The accuracy of the model is:{accuracy} \n\
          \n\n\n ########################################################')

myexecute(f'rsync -Pav {os.path.join(cur_dir, best_model_name)}checkpoint.h5 /hercules/scratch/atya/IsolatedML/models/ ')
myexecute(f'rsync -Pav {os.path.join(cur_dir, best_model_name)}model.h5 /hercules/scratch/atya/IsolatedML/models/ ')
if cur_dir != '/hercules/scratch/atya/IsolatedML/': 
    myexecute(f'rm -rf {cur_dir}')

# Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_{run}.npy')
# Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_{run}.npy')
# Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_{run}.npy')

# Y_train_freq = 1/Y_train
# Y_test_freq = 1/Y_test
# Y_val_freq = 1/Y_val

# # Y_train = 1/Y_train
# # Y_test = 1/Y_test
# # Y_val = 1/Y_val

# Y_train_arg = np.zeros_like(Y_train_freq)
# for i in range(Y_train_freq.shape[0]):
#     Y_train_arg[i] = np.argmin(np.abs(freq_axis - Y_train_freq[i]))

# Y_test_arg = np.zeros_like(Y_test_freq)
# for i in range(Y_test_freq.shape[0]):
#     Y_test_arg[i] = np.argmin(np.abs(freq_axis - Y_test_freq[i]))

# Y_val_arg = np.zeros_like(Y_val_freq)
# for i in range(Y_val_freq.shape[0]):
#     Y_val_arg[i] = np.argmin(np.abs(freq_axis - Y_val_freq[i]))
# #Y_train_arg = np.argmin(np.abs(freq_axis - Y_train_freq[:,0,None]), axis=1)

# Y_train_arg = Y_train_arg.astype(int)
# Y_test_arg = Y_test_arg.astype(int)
# Y_val_arg = Y_val_arg.astype(int)

# Y_train = Y_train_arg.astype(np.float64)
# Y_test = Y_test_arg.astype(np.float64)
# Y_val = Y_val_arg.astype(np.float64)

# fft_length = X_train.shape[1]
# #convert fft length to a power of 2
# fft_length = 2**int(np.log2(fft_length))

# X_train = X_train[:,:fft_length]
# X_test = X_test[:,:fft_length]
# X_val = X_val[:,:fft_length]


# X_val_freq = np.zeros((X_val.shape[0],400,2))
# for i in range(X_val.shape[0]):
#     freq = freq_axis[Y_val_arg[i]]
#     min_ind = np.max([0,Y_val_arg[i]-200])
#     max_ind = min_ind + 400
#     X_val_freq[i,:,0] = X_val[i,min_ind:max_ind]
# #    X_val_freq[i,:,1] = (freq_axis[min_ind:max_ind])
# #    X_val_freq[i,:,1] = (1/freq_axis[min_ind:max_ind])*(1000)
#     X_val_freq[i,:,1] = np.arange(min_ind,max_ind,1).astype(np.float64)
# #    X_val_freq[i,:,1] = np.arange(0,400,1).astype(np.float64)

# X_test_freq = np.zeros((X_test.shape[0],400,2))
# for i in range(X_test.shape[0]):
#     freq = freq_axis[Y_test_arg[i]]
#     min_ind = np.max([0,Y_test_arg[i]-200])
#     max_ind = min_ind + 400
#     X_test_freq[i,:,0] = X_test[i,min_ind:max_ind]
#     #X_test_freq[i,:,1] = (freq_axis[min_ind:max_ind])
#     #X_test_freq[i,:,1] = (1/freq_axis[min_ind:max_ind])*(1000)
#     X_test_freq[i,:,1] = np.arange(min_ind,max_ind,1).astype(np.float64)

# X_train_freq = np.zeros((X_train.shape[0],400,2))
# for i in range(X_train.shape[0]):
#     freq = freq_axis[Y_train_arg[i]]
#     min_ind = np.max([0,Y_train_arg[i]-200])
#     max_ind = min_ind + 400
#     X_train_freq[i,:,0] = X_train[i,min_ind:max_ind]
#     #X_train_freq[i,:,1] = (freq_axis[min_ind:max_ind])
#     #X_train_freq[i,:,1] = (1/freq_axis[min_ind:max_ind])*(1000)
#     X_train_freq[i,:,1] = np.arange(min_ind,max_ind,1).astype(np.float64)

# X_train = X_train_freq
# X_test = X_test_freq
# X_val = X_val_freq

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1],X_test.shape[2], 1) 

# Y_train = (Y_train.reshape(Y_train.shape[0], 1))
# Y_test = (Y_test.reshape(Y_test.shape[0], 1))
# Y_val = (Y_val.reshape(Y_val.shape[0], 1))

import numpy as np
import sys
import pandas as pd

process_type = sys.argv[1]
run = sys.argv[2]
indices = np.load(f'raw_data/{run}/{process_type}_indices.npy')


dat_size = 16777216
time_res = 64e-6 # in seconds
T_obs = (dat_size*time_res)/60 # in minutes is equal to 17.895 minutes
freq_axis = np.fft.rfftfreq(dat_size, d=64e-6)
freq_res = 1/(T_obs*60)

data_array = np.memmap(f'raw_data/{run}/{process_type}_data.npy', dtype=np.float32, mode='w+', shape=(len(indices), dat_size // 2 + 1))
labels_df = pd.read_csv(f'meta_data/labels_{run}.csv')

for i, ind in enumerate(indices):
    file = f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{labels_df["file_name"].values[ind]}'
    with open(file, 'rb') as f:
        dat = np.frombuffer(f.read(), dtype=np.float32)
        dat = dat[:dat_size]
        dat = (dat - np.mean(dat)) / np.std(dat)
        fft = np.fft.rfft(dat)
        power = np.abs(fft) ** 2
        power = (power - np.mean(power)) / np.std(power)
        data_array[i, :] = power
    print(i, 'done')

data_array.flush()
del data_array
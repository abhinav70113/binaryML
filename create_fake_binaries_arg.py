import numpy as np
import glob
import os
from joblib import Parallel, delayed
from time import time
import sys
import pandas as pd
def myexecute(cmd):
    print("'%s'"%cmd)
    os.system("echo '%s'"%cmd)
    os.system(cmd)

run = 'BB'
size = 2000
#not done 18-24
#done  24(1152) -42
start_ind = int(sys.argv[2])
end_ind = int(sys.argv[3])
start = start_ind*48 
end = end_ind*48
if end > size:
    end = size
job_id = sys.argv[1]
current_dir = f'/tmp/Abhinav_DATA{job_id}/sims/'
myexecute(f'mkdir -p /tmp/Abhinav_DATA{job_id}/sims/')

# runBA: first simulation of binaries

# $PSRHOME/psrsoft/usr/src/sixproc/src/fake -period 1-20 -snrpeak 0.3-0.8 -dm 40 -width 10-30 -nbits 8 -tsamp 64 
# -tobs 1080 -nchans 1024 -fch1 2843.75 -foff 0.8544921875 -binary -bper 3-30 -becc 0.0 -binc 0-90 -bpmass 1.4 -bcmass 0.1-1.4 -bphase 0-1 > fake4.fil

# period_array = np.random.uniform(0.001, 0.02, size) #Uniform period in ms
# snr_array = np.random.uniform(0.3, 0.8, size) #signal to noise ratio
# width_array = np.random.uniform(10, 30, size) #width of the pulse in ms
# bper_array = np.random.uniform(3, 30, size) #binary period in hours
# binc_array = np.random.uniform(0, 90, size) #binary inclination angle in degrees
# bcmass_array = np.random.uniform(0.1, 1.4, size) #companion mass in solar masses
# bphase_array = np.random.uniform(0, 1, size) #binary phase


csv = pd.read_csv(f'/hercules/results/atya/BinaryML/sims/run{run}/highSNR_run{run}.csv')
ind_array = csv['# ind'].values[start:end]
period_array = csv['period'].values[start:end]*1000 #Uniform period in ms
width_array = csv['width'].values[start:end]
snr_array = csv['snr'].values[start:end]
bper_array = csv['bper'].values[start:end]
binc_array = csv['binc'].values[start:end]
bcmass_array = csv['bcmass'].values[start:end]
bphase_array = csv['bphase'].values[start:end]

# meta_data = np.zeros((size,8),np.float32)
# meta_data[:,0] = np.arange(0,size,1)
# meta_data[:,1] = period_array
# meta_data[:,2] = snr_array
# meta_data[:,3] = width_array
# meta_data[:,4] = bper_array
# meta_data[:,5] = binc_array
# meta_data[:,6] = bcmass_array
# meta_data[:,7] = bphase_array


#save numpy array as csv file
#np.savetxt(f'/hercules/results/atya/IsolatedML/sims/run{run}/highSNR_run{run}.csv',meta_data,delimiter=','
# ,header='ind,period,snr,width,bper,binc,bcmass,bphase')

def fake(ind,period,snr,width,bper,binc,bcmass,bphase,run):
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    fake_prefix = '/home/psr/software/psrsoft/usr/src/sixproc/src/fake ' # 268.435456
    myexecute(sing_prefix+fake_prefix+f'-period {period} -snrpeak {snr} -width {width} -dm 40 -nbits 8 -tsamp 64 -tobs 1080 -nchans 1024 -fch1 2843.75 -foff 0.8544921875 -binary -bper {bper} -becc 0.0 -binc {binc} -bpmass 1.4 -bcmass {bcmass} -bphase {bphase} > {current_dir}obs{ind}{run}.fil')   
    myexecute(sing_prefix+f'prepdata -dm 40.0 -nobary -o {current_dir}obs{ind}{run} {current_dir}obs{ind}{run}.fil')
    myexecute(f'rm {current_dir}obs{ind}{run}.fil'.format(ind,current_dir,run))
    myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n Number:{ind} done \n\n\n ############################################################################## \n\n\n \"')
    


start_time = time()
val = Parallel(n_jobs=-1)\
            (delayed(fake)\
            (int(ind),period,snr,width,bper,binc,bcmass,bphase,run)\
            for ind,period,snr,width,bper,binc,bcmass,bphase \
            in zip(
                ind_array,
                period_array,
                snr_array,
                width_array,
                bper_array,
                binc_array,
                bcmass_array,
                bphase_array))
print('\n\n####################################################################')
print("Time taken: %s" % (time() - start_time))
print('####################################################################\n\n')
myexecute(f'rsync -Pav /tmp/Abhinav_DATA{job_id}/sims/*.dat /hercules/results/atya/BinaryML/sims/run{run}/ ')
myexecute(f'rsync -Pav /tmp/Abhinav_DATA{job_id}/sims/*.inf /hercules/results/atya/BinaryML/sims/run{run}/ ')
myexecute(f'rm -rf /tmp/Abhinav_DATA{job_id}')

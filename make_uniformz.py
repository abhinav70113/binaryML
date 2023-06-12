import numpy as np
import math
import os
from joblib import Parallel, delayed
# functions to calculate z and a_max
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

def calculate_z(T_obs,a,h,P_s):
    T_obs = T_obs*3600
    c = 299792458
    return T_obs**2*a*h/(P_s*c)  

def calculate_a_from_z(z,T_obs,h,P_s):
    T_obs = T_obs*3600
    c = 299792458
    return z*P_s*c/(T_obs**2*h)

def calculate_P_s(z,T_obs,a,h):
    T_obs = T_obs*3600
    c = 299792458
    return T_obs**2*a*h/(z*c)  

def acc_max(m_c,m_p,i,P_orb):
    T_const = 4.925490947e-6
    c = 299792458
    solar_mass = 1.98847e30
    #m_c = m_c*solar_mass
    #m_p = m_p*solar_mass
    P_orb = P_orb*3600
    f = (m_c*np.sin(i*np.pi/180))**3/(m_c+m_p)**2
    #print(f/solar_mass)
    return (2*np.pi/P_orb)**(4/3)*(T_const*f)**(1/3)*c

def calculate_z_mag_from_pd(period_pos,period_neg,freq_res):
    freq_pos = 1/period_pos
    freq_neg = 1/period_neg
    z = np.abs(freq_pos - freq_neg)/freq_res
    return z

def calculate_pd(p,a):
    '''
    p: period in seconds
    a: acceleration in m/s^2
    '''
    c = 299792458
    return p*a/c

def calculate_presto_fold_p_pos(p,pd,T_obs):
    '''
    p: period in seconds
    pd: period derivative in seconds/second
    T_obs: observation time in hours
    '''
    T_obs = T_obs*3600
    return p + pd*T_obs/2

def calculate_presto_fold_p_neg(p,pd,T_obs):
    '''
    p: period in seconds
    pd: period derivative in seconds/second
    T_obs: observation time in hours
    '''
    T_obs = T_obs*3600
    return p - pd*T_obs/2  

def calculate_a(i,P_orb,m_c,phase,T_obs):
    '''
    i: inclination angle in degrees
    P_orb: orbital period in hours
    m_c: companion mass in solar masses
    P_s: spin period in seconds
    phase: orbital phase from 0 to 1
    T_obs: observation time in hours
    '''
    fake_inclination = i
    p_orb_days = P_orb/24
    fake_orbital_period =  P_orb * 3600
    fake_initial_orbital_phase = phase
    #fake_initial_orbital_phase = .5
    fake_eccentricity = 0.0
    fake_longitude_periastron = 0 * np.pi/180
    mass_companion = m_c
    mass_pulsar = 1.4

    G = 6.67408e-11
    c = 299792458
    M_SUN = 1.98847e30

    observation_time = fake_orbital_period
    n_samples = 10000
    time = np.linspace(0, observation_time, n_samples)

    incl = fake_inclination * (np.pi/180)
    omegaB = 2.0 * np.pi/fake_orbital_period
    t0 = fake_initial_orbital_phase * fake_orbital_period
    massFunction = math.pow((mass_companion * np.sin(incl)), 3)/math.pow((mass_companion + mass_pulsar), 2)
    asini = math.pow(( M_SUN * massFunction * G * fake_orbital_period * \
                    fake_orbital_period / (4.0 * np.pi * np.pi)), 0.333333333333)

    meanAnomaly = omegaB * (time - t0)
    eccentricAnomaly = meanAnomaly + fake_eccentricity * np.sin(meanAnomaly) * \
    (1.0 + fake_eccentricity * np.cos(meanAnomaly))

    du = np.ones(n_samples)
    for i in range(len(du)):
        while(abs(du[i]) > 1.0e-13):
        
            du[i] = (meanAnomaly[i] - (eccentricAnomaly[i] - fake_eccentricity * \
                                np.sin(eccentricAnomaly[i])))/(1.0 - fake_eccentricity * np.cos(eccentricAnomaly[i]))

            eccentricAnomaly[i] += du[i]


    trueAnomaly = 2.0 * np.arctan(math.sqrt((1.0 + fake_eccentricity)/(1.0 - fake_eccentricity)) \
                                * np.tan(eccentricAnomaly/2.0))

    los_velocity = omegaB * (asini / (np.sqrt(1.0 - math.pow(fake_eccentricity, 2)))) * \
            (np.cos(fake_longitude_periastron + trueAnomaly) + fake_eccentricity * np.cos(fake_longitude_periastron))

    los_acceleration = (-1*(omegaB*omegaB)) * (asini / math.pow(1 - math.pow(fake_eccentricity, 2), 2)) * \
    np.power((1 + (fake_eccentricity * np.cos(trueAnomaly))), 2) * np.sin(fake_longitude_periastron + trueAnomaly)

    end = T_obs * 3600
    phase_pi = np.pi * fake_orbital_period/(2*np.pi)
    phase_pi_ind = np.argmin(np.abs(time-phase_pi))
    end_ind = np.argmin(np.abs(time-end))
    los_acc_off = los_acceleration[phase_pi_ind:phase_pi_ind+end_ind] #take the presto convention
    a = np.mean(los_acc_off)
    a_max_incline = acc_max(m_c,1.4,fake_inclination,P_orb)
    return a,a_max_incline,los_velocity,los_acc_off,time,phase_pi_ind,end_ind

def return_params(i,P_orb,m_c,P_s,phase,T_obs,h):
    c = 299792458
    pRest = P_s
    a,a_max_incline,los_velocity,los_acc_off,time,phase_pi_ind,end_ind = calculate_a(i,P_orb,m_c,phase,T_obs)
    pApp = pRest/(1.0 - (los_velocity / c))
    period_new = pApp[end_ind//2]  #taking the period in the middle of the observation as presto does
    z_shifted_pi = calculate_z(T_obs,a,h,period_new)
    z_max_incline = calculate_z(T_obs,a_max_incline,h,period_new)
    print('next')
    return period_new,a,a_max_incline,z_shifted_pi,z_max_incline
#,los_acceleration,los_velocity,pApp,time
fft_size = 16777216
time_res = 64e-6 # in seconds
T_obs = (fft_size*time_res)/60 # in minutes is equal to 17.895 minutes
freq_axis = np.fft.rfftfreq(fft_size, d=64e-6)
freq_res = 1/(T_obs*60)

np.random.seed(42)
#size = 2000
sims = 2000000
#period_array = np.random.uniform(0.001, 0.02, sims) #Uniform period in ms
snr_array = np.random.uniform(0.3, 0.8, sims) #signal to noise ratio
width_array = np.random.uniform(10, 30, sims) #width of the pulse in ms
bper_array = np.random.uniform(3, 30, sims) #binary period in hours
binc_array = np.random.uniform(0, 90, sims) #binary inclination angle in degrees
bcmass_array = np.random.uniform(0.1, 1.4, sims) #companion mass in solar masses
bphase_array = np.random.uniform(0, 1, sims) #binary phase
z_array = np.zeros_like(bphase_array)
period_array = np.zeros_like(bphase_array)
c = 299792458 #speed of light in m/s


def final_z_calc(i):
    params = calculate_a(binc_array[i],bper_array[i],bcmass_array[i],bphase_array[i],T_obs/60) 
    a = params[0]
    vel = params[2]
    end_ind = params[6]
    p = 0
    while (p < 0.001) or (p > 0.02): #period should be in the range of 1-20 ms
        if a < 0:
            z_temp = np.random.uniform(-50,0)
        else:
            z_temp = np.random.uniform(0,50)
        p = calculate_P_s(z_temp,T_obs/60,a,1)*(1-(vel[end_ind//2]/c))
        
    myexecute(f'echo "{i},{p},{snr_array[i]},{width_array[i]},{bper_array[i]},{binc_array[i]},{bcmass_array[i]},{bphase_array[i]},{z_temp}"')
    return i, p, z_temp

results = Parallel(n_jobs=-1)(delayed(final_z_calc)(i) for i in range(sims))
for i, p, z in results:
    period_array[i] = p
    z_array[i] = z 

meta_data = np.zeros((sims,9),np.float32)
meta_data[:,0] = np.arange(0,sims,1)
meta_data[:,1] = period_array
meta_data[:,2] = snr_array
meta_data[:,3] = width_array
meta_data[:,4] = bper_array
meta_data[:,5] = binc_array
meta_data[:,6] = bcmass_array
meta_data[:,7] = bphase_array
meta_data[:,8] = z_array

cur_dir = f'/tmp/Abhinav_DATA_uniform/'
root_dir = '/hercules/scratch/atya/BinaryML/'

myexecute(f'mkdir -p {cur_dir}')

#save numpy array as csv file
np.savetxt(f'{cur_dir}uniformZv2.csv',meta_data,delimiter=',',header='ind,period,snr,width,bper,binc,bcmass,bphase,z')
#rsync the csv file to the hercules scratch directory
myexecute(f'rsync -avz {cur_dir} {root_dir}meta_data/')
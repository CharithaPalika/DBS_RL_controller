import numpy as np
import yaml
from .stn_gpe_weights import interconnectivity,random_wts_sparse
from .utils import load_yaml,lfp_dist_matrix
# from tqdm import tqdm
from tqdm.notebook import tqdm
from .dbs import *

def STN_GPe_loop(exp_yaml_path):

    # Param dict

    arguments = load_yaml(exp_yaml_path)
    # print(arguments)
    I_strd2_gpe = arguments.get('I_strd2_gpe')
    lat_strength_stn = arguments.get('lat_strength_stn')
    lat_strength_gpe = arguments.get('lat_strength_gpe')
    wsg_strength = arguments.get('wsg_strength')
    wgs_strength = arguments.get('wgs_strength')
    I_gpe_ext = arguments.get('I_gpe_ext')
    I_stn_ext = arguments.get('I_stn_ext')
    binsize = arguments.get('binsize')
    

    # DBS parameters
    DBS = arguments.get('DBS')
    dbs_amplitude = arguments.get('dbs_amplitude')
    freq = arguments.get('freq')
    duty = arguments.get('duty')
    sampling_freq = arguments.get("time")
    pulseinterval = arguments.get('pulseinterval')
    # Simulation time
    time = arguments.get("time") # Time for the simulation
    dt = arguments.get("dt") # Time step for Euler method
    stn_gpe_noise = arguments.get('stn_gpe_noise')
    

    # DBS object instantiation
    dbs = GenerateDBS()
    # Parameters

    #Izikevich parameters
    #STN
    a_stn = 0.005
    b_stn = 0.265
    c_stn = -65
    d_stn = 1.5

    #GPe
    a_gpe = 0.1
    b_gpe = 0.2
    c_gpe = -65
    d_gpe = 2

    # Intialising a matrix for number of neurons in each block of STN and GPe
    stn_gpe_units = arguments.get("stn_gpe_units")

    V_stn = np.zeros((stn_gpe_units,stn_gpe_units))
    V_gpe = np.zeros((stn_gpe_units,stn_gpe_units))

    # Intialising recovery variable for STN, GPe, GPi
    U_stn = np.zeros((stn_gpe_units,stn_gpe_units))
    U_gpe = np.zeros((stn_gpe_units, stn_gpe_units))

    # Gating variable initiation
    #GPe
    h_gaba_gpe = np.zeros((stn_gpe_units,stn_gpe_units))
    h_nmda_gpe = np.zeros((stn_gpe_units,stn_gpe_units))
    h_ampa_gpe = np.zeros((stn_gpe_units,stn_gpe_units))
    h_strd2_gpe = np.zeros((stn_gpe_units,stn_gpe_units))

    #STN
    h_gaba_stn = np.zeros((stn_gpe_units,stn_gpe_units))
    h_nmda_stn = np.zeros((stn_gpe_units,stn_gpe_units))
    h_ampa_stn = np.zeros((stn_gpe_units,stn_gpe_units))

    # Intialising spiking activity
    spk_gpe = np.zeros((stn_gpe_units,stn_gpe_units))
    spk_stn = np.zeros((stn_gpe_units,stn_gpe_units))
    spk_d2 = np.ones((stn_gpe_units,stn_gpe_units))

    # Decay constants for receptors (ms)
    tau_gaba = 4
    tau_nmda = 160
    tau_ampa = 6

    # Synaptic voltage of different receptors
    E_gaba = -60
    E_nmda = 0
    E_ampa = 0

    mg = 1 # Magnesium conc

    # D2 weight
    w_strd2_gpe = 1

    # Lateral weights
    lat_sparse = arguments.get('lat_sparse')

    w_lat_stn = random_wts_sparse(lat_strength_stn, lat_sparse,(stn_gpe_units * stn_gpe_units, stn_gpe_units, stn_gpe_units))
    w_lat_gpe = random_wts_sparse(lat_strength_gpe, lat_sparse, (stn_gpe_units * stn_gpe_units,stn_gpe_units,stn_gpe_units))

    # Interconnectivites
    inter_sparse = arguments.get('inter_sparse')
    wsg = random_wts_sparse(wsg_strength, inter_sparse, (stn_gpe_units * stn_gpe_units, stn_gpe_units, stn_gpe_units))
    wgs = random_wts_sparse(wgs_strength, inter_sparse, (stn_gpe_units * stn_gpe_units, stn_gpe_units, stn_gpe_units))

    # LFP distance matrix
    lfp_dist = lfp_dist_matrix(stn_gpe_units,7,7)


    # Peak voltage for izikevich
    vpeak = 30

    # Variable to store all voltages wrt to time
    V_gpe_time = [] # stores voltages for a specific neuron in GPe
    V_stn_time = [] # stores voltages for a specific neuron in stn

    V_gpe_time_all = [] # stores voltages for all neurons in GPe
    V_stn_time_all = [] # stores voltages for all neurons in stn


    lfp_stn = [] # Computes total lfp for STN
    lfp_gpe = [] # Computes total lfp for GPe

    spike_monitor_stn = []
    spike_monitor_gpe = []

    # DBS Spread (Gaussian)
    dbs_spread = dbs.dbs_gauss_weight(n = arguments.get('stn_gpe_units'), 
                                  c = arguments.get('center'), 
                                  amplitude = arguments.get('spread_amplitude'), 
                                  sigma = arguments.get('sigma'))
    
    
    # DBS time signal
    if DBS == True:
        if arguments.get('DBS_func') == 'monophasicDBS':
            I_DBS = dbs.monophasicDBS(amplitude = arguments.get('DBS_amplitude'), 
                                  T = 1/arguments.get('DBS_freq'), 
                                  duty = arguments.get('DBS_duty'), 
                                  sampling_freq = (10**3)/arguments.get('dt'), 
                                  time_sec = arguments.get('time') *arguments.get('dt')/(10**3)) 
            
        if arguments.get('DBS_func') == 'biphasicDBS':
            I_DBS = dbs.biphasicDBS(duty = arguments.get('DBS_duty'),
                                T = 1/arguments.get('DBS_freq'), 
                                A1 = arguments.get('DBS_A1'), 
                                A2 = arguments.get('DBS_A2'),
                                sampling_freq = (10**3)/arguments.get('dt'), 
                                time_sec = arguments.get('time') *arguments.get('dt')/(10**3),
                                pulseinterval = pulseinterval)
            
        if arguments.get('DBS_func') == 'biphasicDBS_uninoise':
            I_DBS = dbs.biphasicDBS_uninoise(duty = arguments.get('DBS_duty'), 
                                         T = 1/arguments.get('DBS_freq'), 
                                         A1 = arguments.get('DBS_A1'), 
                                         A2 = arguments.get('DBS_A2'), 
                                         low = arguments.get('DBS_unilow'), 
                                         high = arguments.get('DBS_unihigh'), 
                                         sampling_freq = (10**3)/arguments.get('dt'), 
                                         time_sec = arguments.get('time') *arguments.get('dt')/(10**3))
            
        if arguments.get('DBS_func') == 'biphasicDBS_normalnoise':
            I_DBS = dbs.biphasicDBS_normalnoise(duty = arguments.get('DBS_duty'), 
                                            T = 1/arguments.get('DBS_freq'), 
                                            A1 = arguments.get('DBS_A1'), 
                                            A2 = arguments.get('DBS_A2'), 
                                            mean = arguments.get('DBS_normmean'), 
                                            std = arguments.get('DBS_normstd'), 
                                            sampling_freq = (10**3)/arguments.get('dt'), 
                                            time_sec = arguments.get('time') *arguments.get('dt')/(10**3))
        if arguments.get('DBS_func') == 'biphasicDBS_noisy':
            I_DBS = dbs.biphasicDBS_noisy(duty = arguments.get('DBS_duty'), 
                                                T = 1/arguments.get('DBS_freq'), 
                                                amplitude = arguments.get('DBS_amplitude'), 
                                                mean = arguments.get('DBS_normmean'), 
                                                std = arguments.get('DBS_normstd'), 
                                                sampling_freq = (10**3)/arguments.get('dt'), 
                                                time_sec = arguments.get('time') *arguments.get('dt')/(10**3))
            
        if arguments.get('DBS_func') == 'biphasicDBS_variablefreq':
            I_DBS = dbs.biphasicDBS_variablefreq(pulse_width_micro  = arguments.get('pulse_width_micro'),
                                                 T = 1/arguments.get('DBS_freq'),
                                                 amplitude = arguments.get('DBS_amplitude'),
                                                 mean = arguments.get('DBS_normmean'),
                                                 std = arguments.get('DBS_normstd'),
                                                 sampling_freq = (10**3)/arguments.get('dt'),
                                                 time_sec= arguments.get('time') *arguments.get('dt')/(10**3))
        

    else:
        I_DBS = np.zeros(sampling_freq)

    for i in tqdm(range(time)):

        dbs_spread_ = dbs_spread

        # SYNAPTIC CURRENT CALCULATION FOR GPe
        #I_gabalat_gpe
        dh_gaba_gpe = (-h_gaba_gpe + spk_gpe)/tau_gaba
        h_gaba_gpe = h_gaba_gpe + dt * dh_gaba_gpe
        I_gabalat_gpe = (np.sum(np.sum((w_lat_gpe * h_gaba_gpe), axis =1), axis = 1).reshape(stn_gpe_units, stn_gpe_units))*(E_gaba - V_gpe)

        # I_nmda_gpe
        B_gpe = 1/(1+(mg/3.57)*np.exp(-0.062 * V_gpe))
        dh_nmda_gpe = (-h_nmda_gpe + spk_stn)/tau_nmda
        h_nmda_gpe = h_nmda_gpe + dt * dh_nmda_gpe
        I_nmda_gpe = (np.sum(np.sum((wsg * h_nmda_gpe), axis =1),axis = 1).reshape(stn_gpe_units, stn_gpe_units))*(E_nmda -V_gpe) * B_gpe

        # I_ampa_gpe
        dh_ampa_gpe = (-h_ampa_gpe + spk_stn)/tau_ampa
        h_ampa_gpe = h_ampa_gpe + dt * dh_ampa_gpe
        I_ampa_gpe = (np.sum(np.sum((wsg * h_ampa_gpe), axis =1),axis = 1).reshape(stn_gpe_units, stn_gpe_units))*(E_ampa - V_gpe)

        
        #dh_strd2_gpe = (-h_strd2_gpe + spk_d2* poisson_spikes[i])/tau_gaba
        #h_strd2_gpe = h_strd2_gpe + dt * dh_strd2_gpe
        #I_strd2_gpe = w_strd2_gpe * h_strd2_gpe *(E_gaba -V_gpe)

        # Total current GPe
        cD2 = -1
        I_gpe_t = I_gabalat_gpe + I_nmda_gpe + I_ampa_gpe + I_strd2_gpe  * cD2

        # Membrane voltage using izikevich neuron
        dV_gpe = (0.04 * V_gpe**2) + 5 * V_gpe - U_gpe + 140 + I_gpe_t + I_gpe_ext + stn_gpe_noise *  np.random.randn(stn_gpe_units, stn_gpe_units)
        dU_gpe = a_gpe *((b_gpe*V_gpe) - U_gpe)
        V_gpe = V_gpe + dt * dV_gpe
        U_gpe = U_gpe + dt * dU_gpe

        #Replacement criteria for izikevich neuron
        indx_gpe = np.where(V_gpe > vpeak)
        V_gpe[indx_gpe] = c_gpe
        U_gpe[indx_gpe] = U_gpe[indx_gpe] + d_gpe
        V_gpe_time.append(V_gpe[5,5])
        V_gpe_time_all.append(V_gpe)
        spk_gpe = np.zeros((stn_gpe_units, stn_gpe_units))
        spk_gpe[indx_gpe] = 1

        spike_monitor_gpe.append(spk_gpe.reshape(stn_gpe_units * stn_gpe_units))

        I_syn_lfp_gpe = np.sum(np.multiply(I_gpe_t, lfp_dist))
        lfp_gpe.append(I_syn_lfp_gpe)


        #SYNAPTIC CURRENT CALCULATION FOR STN

        # I_gaba_stn
        dh_gaba_stn = (-h_gaba_stn + spk_gpe)/tau_gaba
        h_gaba_stn = h_gaba_stn + dt * dh_gaba_stn
        I_gaba_stn = (np.sum(np.sum((wgs * h_gaba_stn), axis =1),axis = 1).reshape(stn_gpe_units, stn_gpe_units))*(E_gaba - V_stn)

        # I_nmdalat
        dh_nmda_stn = (-h_nmda_stn + spk_stn)/tau_nmda
        h_nmda_stn = h_nmda_stn + dt * dh_nmda_stn

        B_stn = 1/(1+(mg/3.57)*np.exp(-0.062 * V_stn))

        temp_nmdalat = h_nmda_stn *(E_nmda - V_stn)
        I_nmdalat_stn = B_stn * (np.sum(np.sum(np.multiply(w_lat_stn, temp_nmdalat), axis =1), axis = 1)).reshape(stn_gpe_units, stn_gpe_units)

        # I_ampalat_stn
        dh_ampa_stn = (-h_ampa_stn + spk_stn)/tau_ampa
        h_ampa_stn = h_ampa_stn + dt * dh_ampa_stn

        temp_ampalat = h_ampa_stn *(E_ampa - V_stn)
        I_ampalat_stn =  (np.sum(np.sum(np.multiply(w_lat_stn, temp_ampalat), axis =1), axis = 1)).reshape(stn_gpe_units, stn_gpe_units)

        
        # Total current STN
        I_stn_t = I_gaba_stn + I_nmdalat_stn + I_ampalat_stn + (I_DBS[i] * dbs_spread_)  + stn_gpe_noise * np.random.randn(stn_gpe_units, stn_gpe_units)

        # Membrane voltage using izikevich neuron
        dV_stn = (0.04 * V_stn**2) + 5 * V_stn - U_stn + 140 + I_stn_t + I_stn_ext
        dU_stn = a_stn *((b_stn*V_stn) - U_stn)
        V_stn = V_stn + dt * dV_stn
        U_stn = U_stn + dt * dU_stn

        #Replacement criteria for izikevich neuron
        indx_stn = np.where(V_stn > vpeak)
        V_stn[indx_stn] = c_stn
        U_stn[indx_stn] = U_stn[indx_stn] + d_stn
        V_stn_time.append(V_stn[5,5])
        V_stn_time_all.append(V_stn)

        spk_stn = np.zeros((stn_gpe_units, stn_gpe_units))
        spk_stn[indx_stn] = 1

        spike_monitor_stn.append(spk_stn.reshape(stn_gpe_units* stn_gpe_units))

        # LFP for stn
        I_syn_lfp_stn = np.sum(np.multiply(I_stn_t - (I_DBS[i] * dbs_spread_), lfp_dist))
        lfp_stn.append(I_syn_lfp_stn)

    results = {'v_stn': V_stn_time_all,
               'v_gpe' : V_gpe_time_all,
               'spike_stn' : spike_monitor_stn,
               'spike_gpe' : spike_monitor_gpe,
               'lfp_stn' : lfp_stn,
               'lfp_gpe' : lfp_gpe,
               'I_DBS' : I_DBS}
    return results
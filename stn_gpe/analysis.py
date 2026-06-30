import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch
from scipy.stats import entropy
import scipy
from numpy.fft import fft, ifft

class Analysis:

    def __init__(self, spike_array):
        self.spike_array = np.array(spike_array)



    def rescale(self, rate_abcd_processed, clip_max = None, clip_min = None, target_mean = 1, max_std = 0.335):
        rate_mean = rate_abcd_processed.mean(axis = 0, keepdims = True)
        rate_std = rate_abcd_processed.std(axis = 0, keepdims = True)
        scale = np.minimum(1.0, max_std / (rate_std + 1e-12))
        rate_rescaled = (rate_abcd_processed - rate_mean) * scale + rate_mean
        rate_rescaled_shifted = rate_rescaled + (target_mean - rate_rescaled.mean(axis = 0, keepdims = True))

        if clip_max is not None or clip_min is not None:
            rate_rescaled_shifted = np.clip(rate_rescaled_shifted, a_min=clip_min, a_max=clip_max)
        return rate_rescaled_shifted
        

    # def rescale(self, stn_abcd_processed, clip_max = 1.75, clip_min = 0.5, target_mean = 1, max_std = 0.35):


        # global_mean = stn_abcd_processed.mean()
        # 
        # stn_abcd_rescaled = stn_abcd_processed + (target_mean - global_mean)

        # stn_abcd_rescaled = np.clip(stn_abcd_rescaled, a_min=clip_min, a_max=clip_max)
        # clipped_std = stn_abcd_rescaled.std()
        # if clipped_std > max_std:
            # print(f'exceeding max std, {clipped_std}')
            # stn_abcd_rescaled = stn_abcd_rescaled * (max_std / (clipped_std + 1e-12))

        # stn_abcd_processed = stn_abcd_processed + (1.0 - global_mean)
        # stn_abcd_rescaled = np.clip(stn_abcd_processed, a_min=clip_min, a_max=clip_max)
        # return stn_abcd_rescaled

    def spike_rate(self,binsize, scaling_factor = 8):
        ''' 
        Function to convert spike to rate of change

        Args:
            spike_array (np.ndarray): Spike train with shape (time, num_neurons)
            binsize (int): binsize
        '''
        num_neurons = self.spike_array.shape[1] 
        grid_size = int(np.sqrt(num_neurons)) # 16self.
        time = self.spike_array.shape[0]
        time_sec = time//10000
        # print(time_sec, time)
        rate_coded = np.zeros((num_neurons, time))
        for neuron in range(num_neurons):
            spikes = self.spike_array[:,neuron]
            for i in range(time):
                rate = np.sum(spikes[i:i+binsize])
                rate_coded[neuron][i] = rate
        rate_coded = rate_coded.reshape(grid_size,grid_size,time)
        mean_rate_a = np.mean(rate_coded[0:grid_size//2,0:grid_size//2,:].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_b = np.mean(rate_coded[grid_size//2:grid_size, 0:grid_size//2,:].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_c = np.mean(rate_coded[0:grid_size//2, grid_size//2:grid_size, :].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_d = np.mean(rate_coded[grid_size//2:grid_size, grid_size//2:grid_size, :].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        
        # computing STD
        rate_abcd = [mean_rate_a, mean_rate_b, mean_rate_c, mean_rate_d]
        rate_abcd = np.array([(i-np.min(i[1000: time - 1000]))/(np.max(i) - np.min(i[1000: time - 1000])) for i in rate_abcd])* scaling_factor #*2
        rate_abcd_processed = np.mean(rate_abcd.reshape(4,time_sec * 100, -1), axis = 2)
        rate_abcd_rescaled = self.rescale(rate_abcd_processed)
        rate_abcd_recaled_std = np.std(rate_abcd_rescaled, axis = 0)
        rate_mean_std = np.mean(rate_abcd_recaled_std)
        rate_max_std = np.max(rate_abcd_recaled_std)
        rate_min_std = np.min(rate_abcd_recaled_std)

        rate_data = {'raw_rate_data': rate_coded,
                     'preprocessed_rate_data': rate_abcd,
                     'processed_rate_data': rate_abcd_processed,
                     'rescaled_rate_data': rate_abcd_rescaled,
                     'mean_std': rate_mean_std,
                     'min_std': rate_min_std,
                     'max_std': rate_max_std}
        return rate_data


    def compute_spike_rate_newfunc(self, bin_size, spike_array_ = None):
        """
        Compute the spike rate for each neuron by binning the spike data.

        Parameters:
        spike_array (numpy.ndarray): 2D array of shape (n_neurons, n_timepoints) containing spike data (0 or 1).
        bin_size (int): Size of the time window to bin the spikes.

        Returns:
        numpy.ndarray: 2D array of spike rates of shape (n_neurons, n_bins).
        """

        if spike_array_ is None:
            spike_array_ = self.spike_array.T
        n_neurons, n_timepoints = spike_array_.shape
        n_bins = n_timepoints // bin_size  # Number of bins

        # Reshape spike array by grouping time points into bins
        binned_spikes = spike_array_[:, :n_bins * bin_size].reshape(n_neurons, n_bins, bin_size)

        # Sum the spikes within each bin and normalize by bin size to get the rate
        spike_rates = binned_spikes.sum(axis=2)

        return spike_rates
    
    def raster_plot(self, h = 0.1, title = 'Raster Plot'):
        '''
        Function to plot spike raster

        Args:
        title (str): Name of the title of the plot
        h (float): step size, Default = 0.1
        spike_array(ndarray) : 2D-NumPy array (time_steps, num_neurons)
        spike array with 0s and 1s, where 1-denoting spike

        Returns: None

        '''
        num_neurons = self.spike_array.shape[1]
        time_steps = self.spike_array.shape[0]
        iter = time_steps 
        t = np.linspace(0,iter*h/1000,iter)
        
        fig = plt.figure(figsize = (15,2))
        for n in range(num_neurons):
            plt.scatter(t, (n+1)*self.spike_array[:,n],color='black', s=0.5)
        #range(time_steps)
        plt.ylim(0.5,256)
        plt.xlabel("t (s)")
        plt.ylabel("# neuron")
        plt.title(title)
        plt.show()

    def frequency(self, dt = 0.1):
        '''
        Function to compute frequency
        Args:
            spike_array : 2D-NumPy array (time_steps, num_neurons)
            dt (float) : dt value used for euler
            spike array with 0s and 1s, where 1-denoting spike

        Returns:
            frequency (float): Mean frequency
            frequency_max (float): Max frequency
            frequency_min (float): Min frequency

        '''
        num_neurons = self.spike_array.shape[1]
        time_steps = self.spike_array.shape[0]

        frequency_avg = 1000* np.sum(self.spike_array)/(num_neurons * time_steps * dt)
        frequency_max = np.max(1000* np.sum(self.spike_array, axis = 0)/(time_steps * dt))
        frequency_min = np.min(1000* np.sum(self.spike_array, axis = 0)/(time_steps * dt))


        return frequency_avg, frequency_max, frequency_min

    def synchrony(self):
        '''
        Function to compute synchrony

        Args:
            spike_array: 2D-NumPy array (time_steps, num_neurons)
            spike array with 0s and 1s, where 1-denoting spike

        Returns:
            Ravg, Rvalue, phi
        '''

        spike_array = np.transpose(self.spike_array)

        (num_neurons, time_steps)= spike_array.shape # rm: num_neurons
        Rvalue=[]
        phi=[]
        phi=3000*np.ones((num_neurons,time_steps))

        for neur in range(num_neurons):
                temp=spike_array[neur,:]
                temptime=np.where(temp==1)[0]
                j=1
                while j<temptime.shape[0]-1:
                    for i in range(temptime[j],temptime[j+1]-1):
                        phi[neur,i]=(2*np.pi*(i-temptime[j]))/(temptime[j+1]-temptime[j])
                    j=j+1

        tempM=np.mean(phi, axis =0)
        a=np.sqrt(-1+0j)
        M=np.exp(a*tempM)
        Rvalue = ((np.sum(np.exp(a*phi),axis =0 )/num_neurons))/M
        Ravg = np.mean(abs(Rvalue))
        return Rvalue, Ravg
    
    # def spectral_entropy(self,signal, fs=1.0, nperseg=1, fmax=50, normalize=True):
    #     # Compute the power spectral density (PSD)
    #     freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        
    #     # Mask the PSD to limit the frequency range to 0-100 Hz
    #     mask = (freqs >= 0) & (freqs <= fmax)
    #     freqs = freqs[mask]
    #     psd = psd[mask]
        
    #     # Normalize the PSD to form a probability distribution
    #     psd /= np.sum(psd)
    #     psd = np.where(psd == 0, 1e-12, psd)  # Avoid log(0)
        
    #     # Compute the spectral entropy (Shannon entropy)
    #     se = entropy(psd, base = 2)
        
    #     if normalize:
    #         se /= np.log(len(psd))  # Normalize to [0, 1]
    #     return se

    # def spectral_entropy(self, signal, fs=1.0, nperseg=None, fmax=None, normalize=True):
    #     if nperseg is None:
    #         nperseg = min(256, len(signal))

    #     # Compute the Power Spectral Density (PSD) using Welch's method
    #     freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    #     if fmax is not None:
    #         mask = (freqs >= 0) & (freqs <= fmax)
    #         freqs = freqs[mask]
    #         psd = psd[mask]
    #     psd_normalized = psd / np.sum(psd)

    #     # Calculate Shannon entropy from the normalized PSD
    #     # Filter out zero values to avoid log(0)
    #     psd_positive = psd_normalized[psd_normalized > 0]
    #     spectral_entropy_value = entropy(psd_positive, base=2)

    #     if normalize:
    #         max_entropy = np.log2(len(psd_positive))
    #         if max_entropy == 0:  # Handle cases where psd_positive has only one element
    #             return 0.0
    #         spectral_entropy_value /= max_entropy

    #     return spectral_entropy_value

    def spectral_entropy(self,signal, fs=1.0, nperseg=1, fmax=50, normalize=True):
        # Compute the power spectral density (PSD)
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        
        # Mask the PSD to limit the frequency range to 0-100 Hz
        mask = (freqs >= 0) & (freqs <= fmax)
        freqs = freqs[mask]
        psd = psd[mask]
        
        # Normalize the PSD to form a probability distribution
        psd /= np.sum(psd)
        psd = np.where(psd == 0, 1e-12, psd)  # Avoid log(0)
        
        # Compute the spectral entropy (Shannon entropy)
        se = entropy(psd)
        
        if normalize:
            se /= np.log(len(psd))  # Normalize to [0, 1]
        return se

    def power_beta(self,signal, fs = 10000):
        signal_smooth = scipy.signal.savgol_filter(signal, window_length=11, polyorder = 5, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
        fft_output = fft(signal_smooth)
        N = len(fft_output)
        n = np.arange(N)
        T = N/fs
        freq = n/T
        singal_power =np.abs(fft_output)**2 
        def band_power_linear(power_W, freq, f_low, f_high):
            idx_band = np.where((freq >= f_low) & (freq < f_high))
            avg_power_linear = np.mean(power_W[idx_band])
            avg_power_dB = 10 * np.log10(avg_power_linear + 1e-12)
            return avg_power_dB
        
        power_freq1 = band_power_linear(singal_power, freq, 8.5, 16.5)
        power_freq2 = band_power_linear(singal_power, freq, 19.5, 27.5)
        power_full_beta = band_power_linear(singal_power, freq, 12.5, 30.5)
        return power_freq1, power_freq2, power_full_beta
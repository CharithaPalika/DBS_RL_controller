import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import signal

class GenerateDBS:
  def __init__(self):
     pass
  
  def dbs_gauss_weight(self, n,c,amplitude, sigma):
      '''
      Function for gaussian distribution wts of DBS pulse

      Args:
          n(int): Grid size
          c (int): center
          amplitude (float): Amplitude
          sigma (float): spread of the gaussian pulse

      Returns:
          wt (np.ndarray): gaussian wt matrix
          (* (1/(2* np.pi * sigma**2)))
      '''
      wt = np.zeros((n,n))
      for i in range(n):
          for j in range(n):
              wt[i][j] = amplitude * np.exp(-((i-c)**2 + (j-c)**2)/(2 * sigma**2))

      return wt

  def monophasicDBS(self,amplitude: float, T: float, duty: float, sampling_freq : int, time_sec: float):
      '''
      Function to generate DBS pulse
      Args:
          amplitude (float): amplitude
          freq (float): frequency
          duty (float): duty
          sampling_freq (int): Number of samples per sec
          time_sec (float): total time in sec
      '''
      freq = 1/T
      t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
      dbs_signal = amplitude * signal.square(2 * np.pi * freq * t, duty = duty)
      #dbs_signal[dbs_signal == -1 * amplitude] = -5
      return dbs_signal

  def biphasicDBS(self, duty: float,T:float, A1:float, A2:float,  sampling_freq : int, time_sec: float, pulseinterval : float):
    """
    Generates a biphasic square pulse with variance in T (Uniform noise).

    Args:
      duty (float): duty cycle. duty must be in the interval of [0,1]
      T (float): Pulse period.
      A1 (float): Amplitude of the first phase.
      A2 (float): Amplitude of the second phase.
      sampling_freq (int): Number of samples per sec
      time_sec (float): total time in sec

    Returns:
      pulse: Biphasic square pulse waveform.
    """
    pulse_width = duty * T
    t = np.linspace(0, time_sec, int(time_sec * sampling_freq))
    pulse1 = signal.square(2*np.pi*t/T, duty=duty/2)
    pulse2 = signal.square(2*np.pi*(t + pulseinterval * pulse_width/2)/T, duty=duty/2)
    pulse = A2/2 * pulse1 + A1/2 *  pulse2
    return pulse


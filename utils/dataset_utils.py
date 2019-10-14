import numpy as np
import sys
import glob
from scipy.io import wavfile
# take a h5py as input, return two tff.simulation.ClientData instance.
from python_speech_features import mfcc
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal

import numpy as np
#import librosa
import glob
import os
from tensorflow_federated.python.simulation import hdf5_client_data
sys.path.append('../')
from constant import *
def load_dataset(filepath):
    client_data = hdf5_client_data.HDF5ClientData(filepath)
    #test_client_data = hdf5_client_data.HDF5ClientData(filepath_test)
    return client_data

def generate_spectrogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10, log=True):
    '''
    Generates a spectrogram based on the input audio signal.

    :params:
        audio - numpy array, raw audio signal (example: in wav format)
        sample_rate - Integer
        window_size - Integer, number of milliseconds that we will consider at the time
        step_size - Integer, number of milliseconds (points) to move the kernel,
                    if the step_size == window_size there is no overlapping beteween signal segments
        eps - Integer, very small number used to help calculating log values of the spectrogram
        log - Boolean, if True, this function will return log values of the specgora

    :returns:
        freq - array of sample frequencies.
        times - array of segment times.
        spec - numpy matrix, spectrogram of the input audio signal
    '''
    
    #Calculates the length of each segment
    nperseg = int(round(window_size * sample_rate / 1e3))
    #Calculates the number of points to overlap between segments
    noverlap = int(round(step_size * sample_rate / 1e3))
    
    #Computes spectrogram
    freqs, times, spec = signal.spectrogram(audio, 
                                            fs=sample_rate, 
                                            window='hann', 
                                            nperseg=nperseg, 
                                            noverlap=noverlap, 
                                            detrend=False)
    
    if log:
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)
    
    return freqs, times, spec

def extract_feature(file_list):
    steps = 10
    sample_rate = 16000
    window_size = 20
    arr = np.zeros((len(file_list), 99, 161))
    for i in range(len(file_list)):
        sr, audio = wavfile.read(file_list[i])
        if audio.shape[0] < sample_rate:
            audio = np.append(audio, np.zeros(sample_rate - audio.shape[0]))
        feature = generate_spectrogram(audio, sample_rate = sr, step_size = 10, window_size = 20)[-1]
        arr[i]  = feature
    return arr
def main():
    wavs = glob.glob(DATA_PATH + '/bed/*.wav')
    res = extract_feature(wavs)
    print(res.shape)
    print(res[0])
if __name__ == '__main__':
    main()

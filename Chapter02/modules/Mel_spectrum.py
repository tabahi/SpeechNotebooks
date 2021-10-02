"""
-----
Author: Abdul Rehman
License:  The MIT License (MIT)
Copyright (c) 2020, Tabahi Abdul Rehman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
#from scipy import signal as signallib
#from numba import jit #install numba to speed up the execution
#from wavio import read as wavio_read

from pydub import AudioSegment
from pydub.utils import get_array_type
import array
import matplotlib.pyplot as plt

def frame_segmentation(signal, sample_rate, window_length=0.040, window_step=0.020):

    #Framing
    frame_length, frame_step = window_length * sample_rate, window_step * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    if(num_frames < 1):
        raise Exception("Clip length is too short. It should be atleast " + str(window_length*2)+ " frames")

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    #Hamming Window
    frames *= np.hamming(frame_length)
    #frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    #print (frames.shape)
    return frames, signal_length



def get_filter_banks(frames, sample_rate, f0_min=60, f0_max=4000, num_filt=128, amp_DB=1, norm=0):
    '''
    Fourier-Transform and Power Spectrum

    return filter_banks, hz_points

    filter_banks: array-like, shape = [n_frames, num_filt]

    hz_points: array-like, shape = [num_filt], center frequency of mel-filters
    '''

    NFFT = num_filt*32      #FFT bins (equally spaced - Unlike mel filter)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    #Filter Banks
    nfilt = num_filt
    low_freq_mel = (2595 * np.log10(1 + (f0_min) / 700))
    high_freq_mel = (2595 * np.log10(1 + (f0_max) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    n_overlap = int(np.floor(NFFT / 2 + 1))
    fbank = np.zeros((nfilt, n_overlap))
    
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    if(amp_DB):
        filter_banks = 20 * np.log10(filter_banks)  # dB
    if(norm):
        filter_banks -= (np.mean(filter_banks)) #normalize if norm=1

    return filter_banks, hz_points





def detect_leading_silence(sound, silence_threshold=-100.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

    





def Extract_mel_spec(wav_file_path, window_length=0.025, window_step=0.010, num_filt=256, emphasize_ratio=0.95, norm=0, amp_DB=1, f0_min=30, f0_max=4000, trim=1, fixed_size=0):
    '''
    Parameters
    ----------

    `wav_file_path`: string, Path of the input wav audio file;

    `window_length`: float, optional (default=0.025). Frame window size in seconds;

    `window_step`: float, optional (default=0.010). Frame window step size in seconds;

    `emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies);

    `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

    `f0_min`: int, optional, (default=30), Hertz;

    `f0_max`: int, optional, (default=4000), Hertz;
    

    Returns
    -------
    returns `frames_filter_banks, frame_count, signal_length, trimmed_length`

    `frames_filter_banks`: array-like, mel-spectrum array

    `frame_count`: int, number of filled frames (out of max_frames);

    `signal_length`: float, signal length in seconds;

    `trimmed_length`: float, trimmed length in seconds, silence at the begining and end of the input signal is trimmed before processing;
    '''

    # using pydub
    sound = AudioSegment.from_file(wav_file_path, "wav")
    
    if (trim):
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)    
        sound = sound[start_trim:duration-end_trim]

    change_in_dBFS = -20.0 - sound.dBFS
    
    normalized_sound =  sound.apply_gain(change_in_dBFS)
    normalized_sound.frame_rate
    #raw_signal = normalized_sound.raw_data
    sample_rate = normalized_sound.frame_rate

    samples = normalized_sound.get_array_of_samples()
    raw_signal = np.array(samples)

    
    ''' using wavio
    wav_data = wavio_read(wav_file_path)
    raw_signal = wav_data.data
    sample_rate = wav_data.rate
    '''

    #emphasize_ratio = 0.70
    signal_to_plot = np.append(raw_signal[0], raw_signal[1:] - emphasize_ratio * raw_signal[:-1])
    #signal_to_plot = raw_signal
    
    num_filt = 256
    frames, signal_length = frame_segmentation(signal_to_plot, sample_rate, window_length=window_length, window_step=window_step)
    frames_filter_banks, hz_points = get_filter_banks(frames, sample_rate, f0_min=f0_min, f0_max=f0_max, num_filt=num_filt, amp_DB=amp_DB, norm=norm)
    
    #x-axis points for triangular mel filter used
    #hz_bins_min = hz_points[0:num_filt] #discarding last 2 points
    ##  hz_bins_mid = hz_points[1:num_filt+1] #discarding 1st and last point
    #hz_bins_max = hz_points[2:num_filt+2] #discarding first 2 points

    
    num_of_frames = frames_filter_banks.shape[0]

    #min_peaks_count = 2
    
    neighboring_frames = 2  #number of neighboring frames to compares
    if(num_of_frames < ((neighboring_frames*2)+1)):
        raise Exception("Not enough frames to compare harmonics. Need at least" + str(neighboring_frames*2)+ " frames. Frame count:", str(num_of_frames))
    
    fixed_windows_list = []
    if(fixed_size > 0):
        
        filt_n = frames_filter_banks.shape[1]
        while(frames_filter_banks.shape[0] > fixed_size):
            new_frames = np.zeros((fixed_size, filt_n))
            new_frames[0:fixed_size] = frames_filter_banks[0:fixed_size]
            frames_filter_banks = frames_filter_banks[int(fixed_size-(fixed_size/2)):]
            fixed_windows_list.append(new_frames)
        if(frames_filter_banks.shape[0] <= fixed_size):
            new_frames = np.zeros((fixed_size, filt_n))
            new_frames[0:frames_filter_banks.shape[0]] = frames_filter_banks
            fixed_windows_list.append(new_frames)
        fixed_windows_list = np.array(fixed_windows_list)
    else:
        fixed_windows_list = frames_filter_banks
    

    return fixed_windows_list, signal_length/sample_rate


def save_mel_spectrum(filter_banks, filepath, size_w=3, size_h=3):
    #Plot the mel-spectrogram
    fig = plt.figure(frameon=False)
    fig.set_size_inches(size_w, size_h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.transpose(filter_banks), interpolation=None, cmap='turbo')
    fig.savefig(filepath)
    plt.close()
    #plt.show()


def plot_mel_spectrum_2(filter_banks):
    #Plot the mel-spectrogram
    
    plt.imshow(np.transpose(filter_banks), interpolation=None, cmap='tab20_r')
    plt.xlabel('Frame index')
    plt.ylabel('Mel-Filter Bank')
    plt.rcParams["figure.figsize"] = (24,4)
    plt.show()


#Ignore this function, Currently doesn't work
def Extract_files_formant_features(array_of_clips, features_save_file, window_length=0.025, window_step=0.010, emphasize_ratio=0.7, norm=0, f0_min=30, f0_max=4000, max_frames=400, formants=3,):
    '''
    Parameters
    ----------
    `array_of_clips`: list of Clip_file_Class objects from 'SER_DB.py';

    `features_save_file`: string, Path for HDF file where extracted features will be stored;

    `window_length`: float, optional (default=0.025). Frame window size in seconds;

    `window_step`: float, optional (default=0.010). Frame window step size in seconds;

    `emphasize_ratio`: float, optional (default=0.7). Amplitude increasing factor for pre-emphasis of higher frequencies (high frequencies * emphasize_ratio = balanced amplitude as low frequencies);

    `norm`: int, optional, (default=0), Enable or disable normalization of Mel-filters;

    `f0_min`: int, optional, (default=30), Hertz;

    `f0_max`: int, optional, (default=4000), Hertz;
    
    `max_frames`: int, optional (default=400). Cut off size for the number of frames per clip. It is used to standardize the size of clips during processing.
    
    `formants`: int, optional (default=3). Number of formants to extract;

    returns processed_clips
    ----------------------

    processed_clips: int, number of successfully processing clips;
    '''

    import os
    if(os.path.isfile(features_save_file)):
        print("Removing HDF")
        os.remove(features_save_file)


    total_clips = len(array_of_clips)
    processed_clips = 0
    
    import h5py
    with h5py.File(features_save_file, 'w') as hf:
        dset_label = hf.create_dataset('labels', (total_clips, 11),  dtype='u2')
        dset_features = hf.create_dataset('features', (total_clips, max_frames, formants*4), dtype='u2')
        
        print("Clip", "i", "of", "Total", "SpeakerID", "Accent", "Sex", "Emotion")
        for index, clip in enumerate(array_of_clips):
            try:
                print("Clip ", index+1, "of", total_clips, clip.speaker_id, clip.accent, clip.sex, clip.emotion)
                array_frames_by_features = np.zeros((max_frames, formants*4), dtype=np.uint16)
                #print(clip.filepath)
                array_frames_by_features, frame_count, signal_length, trimmed_length = Extract_wav_file_formants(clip.filepath, window_length, window_step, emphasize_ratio, norm, f0_min, f0_max, max_frames, formants)
                clipfile_size = int(os.path.getsize(clip.filepath)/1000)

                dset_features[index] = array_frames_by_features
                dset_label[index] = [clip.speaker_id, clip.accent, ord(clip.sex), ord(clip.emotion), int(clip.intensity), int(clip.statement), int(clip.repetition), int(frame_count), int(signal_length*1000), int(trimmed_length*1000), clipfile_size]
                processed_clips += 1
            except Exception as e:
                print (e)
            
        print("Read features of", total_clips, "clips")
    
    print("Closing HDF")
    return processed_clips




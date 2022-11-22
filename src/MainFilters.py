# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:02:12 2022

@author: frederich
"""

import librosa

import numpy as np
import scipy.signal as sig
import soundfile as sf
import Filters as jdfilters

def apply_band_pass_filter(file_name, duration_in, from_value, to_value):
    """
    Sample application of bandpass and bandstop
    filters: filter sweep.
    """
    seconds = duration_in
    sound_name = file_name
    print('Filtrando...', file_name)
    y, sr = librosa.load(sound_name, mono = True, duration = seconds)
    fs = sr
    
    # Parameters
    length_seconds = seconds
    length_samples = fs * length_seconds
    Q = 3

    # We have a separate center frequency for each sample
    center_frequency = np.geomspace(from_value, to_value, length_samples)

   
    # The input signal
    #noise = np.random.default_rng().uniform(-1, 1, (length_samples,))
    noise = y
    
    # Actual filtering
    #bandstop_filtered_noise = bandstop_bandpass_filter(noise, Q, center_frequency, fs)
    bandpass_filtered_noise = jdfilters.bandstop_bandpass_filter(noise, Q, center_frequency, fs, 
                                                                        bandpass=True)
    
    # Make the audio files not too loud
    amplitude = 1
    #bandstop_filtered_noise *= amplitude
    bandpass_filtered_noise *= amplitude
    
    # Apply the fade-in and the fade-out to avoid clicks
    #bandstop_filtered_noise = apply_fade(bandstop_filtered_noise)
    bandpass_filtered_noise = jdfilters.apply_fade(bandpass_filtered_noise)
    
    # Write the output audio file #flac
    #sf.write('bandstop.wav', bandstop_filtered_noise, fs)
    final_file_name = 'filtered_'+file_name
    sf.write(final_file_name, bandpass_filtered_noise, fs)
    #print('Fin de filtrando... ', final_file_name)
    return final_file_name
if __name__=='__main__':
    file_name = '21_11_2022_test_21_11_2022 15_13_06 - 15_13_35.wav'
    audio_duration_in_seconds = 30
    from_in_hertz = 7000
    to_in_hertz = 15000
    apply_band_pass_filter(file_name, audio_duration_in_seconds, from_in_hertz, to_in_hertz)
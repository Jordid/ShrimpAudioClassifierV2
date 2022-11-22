# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:07:43 2022

@author: frederich
"""

import csv
import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# model
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
import os
from google.cloud import storage
import time
import librosa
from datetime import datetime

import scipy.signal as sig
import soundfile as sf

df = pd.read_csv('data_v2.csv')

class_list = df.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(class_list)
#print("y: ", y)

input_parameters = df.iloc[:, 1:27]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(input_parameters))
#print("X:", X)

header_test = "filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean \
        spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean \
        mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var".split()

seconds = 30

#Filters fuctions.
def apply_fade(signal):
    """Apply a fade-in and a fade-out to the 1-dimensional signal"""
    # Use a half-cosine window
    window = sig.hann(8192)
    # Use just the half of it
    fade_length = window.shape[0] // 2
    # Fade-in
    signal[:fade_length] *= window[:fade_length]
    # Fade-out
    signal[-fade_length:] *= window[fade_length:]
    # Return the modified signal
    return signal


def second_order_allpass_filter(break_frequency, BW, fs):
    """
    Returns b, a: numerator and denominator coefficients
    of the second-order allpass filter respectively

    Refer to scipy.signal.lfilter for the explanation
    of b and a arrays.

    The coefficients come from the transfer function of
    the allpass (Equation 2 in the article).

    Parameters
    ----------
    break_frequency : number
        break frequency of the allpass in Hz
    BW : number
        bandwidth of the allpass in Hz
    fs : number
        sampling rate in hz

    Returns
    -------
    b, a : array_like
        numerator and denominator coefficients of
        the second-order allpass filter respectively
    """
    tan = np.tan(np.pi * BW / fs)
    c = (tan - 1) / (tan + 1)
    d = - np.cos(2 * np.pi * break_frequency / fs)
    
    b = [-c, d * (1 - c), 1]
    a = [1, d * (1 - c), -c]
    
    return b, a


def bandstop_bandpass_filter(input_signal, Q, center_frequency, fs, bandpass=False):
    """Filter the given input signal

    Parameters
    ----------
    input_signal : array_like
        1-dimensional audio signal
    Q : float
        the Q-factor of the filter
    center_frequency : array_like
        the center frequency of the filter in Hz
        for each sample of the input
    fs : number
        sampling rate in Hz
    bandpass : bool, optional
        perform bandpass filtering if True, 
        bandstop filtering otherwise, by default False

    Returns
    -------
    array_like
        filtered input_signal according to the parameters
    """
    # For storing the allpass output
    allpass_filtered = np.zeros_like(input_signal)
    
    # Initialize filter's buffers
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    
    # Process the input signal with the allpass
    for i in range(input_signal.shape[0]):
        # Calculate the bandwidth from Q and center frequency
        BW = center_frequency[i] / Q
        
        # Get the allpass coefficients
        b, a = second_order_allpass_filter(center_frequency[i], BW, fs)
        
        x = input_signal[i]
        
        # Actual allpass filtering:
        # difference equation of the second-order allpass
        y = b[0] * x + b[1] * x1 +  b[2] * x2 - a[1] * y1 - a[2] * y2
        
        # Update the filter's buffers
        y2 = y1
        y1 = y
        x2 = x1
        x1 = x
        
        # Assign the resulting sample to the output array
        allpass_filtered[i] = y
    
    # Should we bandstop- or bandpass-filter?
    sign = -1 if bandpass else 1
    
    # Final summation and scaling (to avoid clipping)
    output = 0.5 * (input_signal + sign * allpass_filtered)

    return output

#Apply bandpass filter.
def apply_band_pass_filter(source_path, file_name, duration_in, from_value, to_value):
    """
    Sample application of bandpass and bandstop
    filters: filter sweep.
    """
    seconds = duration_in
    sound_name = f"{source_path}{file_name}"
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
    bandpass_filtered_noise = bandstop_bandpass_filter(noise, Q, center_frequency, fs, 
                                                                        bandpass=True)
    
    # Make the audio files not too loud
    amplitude = 1
    #bandstop_filtered_noise *= amplitude
    bandpass_filtered_noise *= amplitude
    
    # Apply the fade-in and the fade-out to avoid clicks
    #bandstop_filtered_noise = apply_fade(bandstop_filtered_noise)
    bandpass_filtered_noise = apply_fade(bandpass_filtered_noise)
    
    # Write the output audio file #flac
    #sf.write('bandstop.wav', bandstop_filtered_noise, fs)
        
    file_name = 'filtered_'+file_name
    final_file_name = f"{source_path}{file_name}"

    print('Filtered: ', final_file_name)
    sf.write(final_file_name, bandpass_filtered_noise, fs)
    #print('Fin de filtrando... ', final_file_name)
    return final_file_name

def analize_audio_model_v2(source_path, sound_name):
    data_test_v2 = 'data_test_v2.csv'
    file = open(data_test_v2, 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_test)

    audio_duration_in_seconds = 30
    from_in_hertz = 4000
    to_in_hertz = 16000    
    
    sound_name = apply_band_pass_filter(source_path, sound_name, audio_duration_in_seconds, from_in_hertz, to_in_hertz)

    filename = sound_name

    print("Start Feature extraction model v2: ", sound_name)
    y, sr = librosa.load(sound_name, mono = True, duration = seconds)
    chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
    rmse = librosa.feature.rms(y = y)
    spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
    spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
    rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

    for e in mfcc:
        to_append += f' {np.mean(e)}'

    file = open(data_test_v2, 'a', newline = '')

    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    #print("End Feature extraction: ", sound_name)

    df_test = pd.read_csv(data_test_v2)
    #df_test.head()

    X_test = scaler.transform(np.array(df_test.iloc[:, 1:27]))
    #print("X_test:", X_test)

    model = tf.keras.models.load_model('models/audio_classifier_model_v2')
    #model.summary()

    # generate predictions for samples
    predictions = model.predict(X_test)
    #print(predictions)

    # generate argmax for predictions
    classes = np.argmax(predictions, axis = 1)
    #print("classes: ", classes)

    # Transform classes number into classes name.
    result = encoder.inverse_transform(classes)

    os.remove(data_test_v2)
    return result[0]
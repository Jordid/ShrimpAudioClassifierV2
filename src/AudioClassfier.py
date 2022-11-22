# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:29:01 2022

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
import Model2 as jdm2

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'shrimpaudios-d0305039b317.json'
storage_client = storage.Client()
bucket_name = 'recorded_audios'

# Model v1.
df = pd.read_csv('data.csv')

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

def build_file_name():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y")
    dt_string = dt_string.replace('/','_')
    dt_string = dt_string.replace(':','_')
    dt_string = dt_string
    return 'Results '+dt_string+'.txt'

def getCurrentDateTime():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = dt_string.replace('/','_')
    dt_string = dt_string.replace(':','_')
    dt_string = dt_string
    return dt_string

def get_date_from_audio_file_name(blob_name):
  folder_name_destino = ''
  if blob_name:
    first_splitted = blob_name.split('/')
    if first_splitted and len(first_splitted) == 2:
      file_name = first_splitted[1]
      second_split = file_name.split()
      if second_split and len(second_split) > 0:
        date = second_split[0]
        date = date.replace("test_", '')
        folder_name_destino = date
  return folder_name_destino

def save_result(message, filename):
    #print('')
    #print('filename: ', filename)
    text_file_name = get_date_from_audio_file_name(filename)
    #print('text_file_name: ', text_file_name)
    text_file_name = 'Results '+text_file_name+'.txt'
    #print('text_file_name: ', text_file_name)
    try:
        file = open('test/Results/'+text_file_name, 'a')
        file.write(message + '\n')
        file.close()
    except:
        print('Error saving result.')
    


def analize_audio_model_v1(sound_name):
    data_test_v1 = 'data_test.csv'
    file = open(data_test_v1, 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_test)

    filename = sound_name

    print("Start Feature extraction model v1: ", sound_name)
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

    file = open(data_test_v1, 'a', newline = '')

    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    #print("End Feature extraction: ", sound_name)

    df_test = pd.read_csv(data_test_v1)
    #df_test.head()

    X_test = scaler.transform(np.array(df_test.iloc[:, 1:27]))
    #print("X_test:", X_test)

    model = tf.keras.models.load_model('models/audio_classifier_model_v1')
    #model.summary()

    # generate predictions for samples
    predictions = model.predict(X_test)
    #print(predictions)

    # generate argmax for predictions
    classes = np.argmax(predictions, axis = 1)
    #print("classes: ", classes)

    # Transform classes number into classes name.
    result = encoder.inverse_transform(classes)

    os.remove(data_test_v1)
    return result[0]



def process_audio_file(blob):
    #print('blob: ', blob.name)
    if blob.name.startswith('test/t'):
        #print('Start download: ',blob.name)
        location = blob.name
        #print('location: ', location)
        blob.download_to_filename(location)
        #print('Finish download: ',location)
        result_model_v1 = analize_audio_model_v1(location)
        
        splitted = location.split('/')
        part_0 = splitted[0]
        part_1 = splitted[1]
        source_path = part_0 + '/'
        result_model_v2 = jdm2.analize_audio_model_v2(source_path, part_1)

        message = location + ' => ' + 'Model_V1: ' + result_model_v1 + ' ' +'Model_V2: ' + result_model_v2

        print('')
        print(message)
        save_result(message, location)

        #try:
        destino_blob = blob.name.replace("test/", "tested/")
        #print('Moviendo: ', destino_blob)
        
        bucket.rename_blob(blob, destino_blob)
        #blob.delete()
        #print('location_2   : ', location)
        
        os.remove(location)
        
        filtered_file_name = source_path + 'filtered_'+part_1
        
        #print('filtered_file_name   : ', filtered_file_name)

        os.remove(filtered_file_name)
        #except:
            #print('Error al mover el archivo.')

print('')
print('Starting classifier...')
print('')
while True:
    #try:
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix = 'test/');
    #print('blobs: ', blobs)
    if blobs:
        for blob in blobs:
            process_audio_file(blob)
    else:
        print('No hay audios para procesar.')
    #except:
     #   print('Error en el proceso de clasificacion.')
      #  time.sleep(0.3)

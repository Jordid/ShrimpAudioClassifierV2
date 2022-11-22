from flask import Flask, request, flash, redirect
from config import config

import csv
import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# model
import tensorflow as tf
import os
from google.cloud import storage
import time
import librosa
from datetime import datetime

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'shrimpaudios-d0305039b317.json'
storage_client = storage.Client()
bucket_name = 'recorded_audios'

df = pd.read_csv('data.csv')

class_list = df.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(class_list)

input_parameters = df.iloc[:, 1:27]
scaler = StandardScaler()
X = scaler.fit_transform(np.array(input_parameters))

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
    return dt_string

def save_result(message):
    text_file_name = build_file_name()
    text_file_name = 'Results '+ text_file_name + '.txt'
    try:
      file = open('test/Results/'+text_file_name, 'a')
      file.write(message + '\n')
      file.close()
    except:
      print('Error saving result.')

def analize_audio(sound_name):
    file = open('data_test.csv', 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_test)

    filename = sound_name

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

    file = open('data_test.csv', 'a', newline = '')

    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

    df_test = pd.read_csv('data_test.csv')

    X_test = scaler.transform(np.array(df_test.iloc[:, 1:27]))

    model = tf.keras.models.load_model('models/audio_classifier_model')

    # generate predictions for samples
    predictions = model.predict(X_test)

    # generate argmax for predictions
    classes = np.argmax(predictions, axis = 1)

    # transform classes number into classes name
    result = encoder.inverse_transform(classes)
    result = result[0]
    message = filename + ' => ' + result
    save_result(message)
    os.remove('data_test.csv')
    return result

def upload_to_bucket(blob_name, file_path, bucket_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    return

app = Flask(__name__)

@app.route('/proccessAudio', methods=['POST'])
def process_audio():
  print('Start, request.......: ')
  try:
    if request.method == 'POST':

      # check if the post request has the file part
      if 'file' not in request.files:
         return 'No file part.'

      file = request.files['file']
      filename = file.filename

      if filename and filename.endswith('.wav'):
        complete_path = 'assets/audios/'+filename
        file.save(complete_path)

        result = analize_audio(complete_path)

        text_file_name = build_file_name()

        location_to_upload = text_file_name+'/'+filename

        print('location_to_upload: ', location_to_upload)
       # print('complete_path: ', complete_path)
       # print('bucket_name: ', bucket_name)
        upload_to_bucket(location_to_upload, complete_path, bucket_name)
        os.remove(complete_path)

        return result
  except Exception as ex:
    print('Error: ', ex)
    return 'Error while procession audio file.'

if __name__ == '__main__':
  #app.config.from_object(config['development'])
  app.run(host='0.0.0.0', port=80)
  #app.run()

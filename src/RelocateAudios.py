import os
from google.cloud import storage
import time

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'shrimpaudios-d0305039b317.json'
storage_client = storage.Client()
bucket_name = 'recorded_audios'

def build_folder_name_destino(blob_name):
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

def relocate_blobs(blob):
  if blob and blob.name.startswith('tested/test') and blob.name.endswith('.wav'):
      folder_name_destino = build_folder_name_destino(blob.name)
      if folder_name_destino and len(folder_name_destino) > 0:
        folder_name_destino = folder_name_destino + '/'
        destino_blob = blob.name.replace("tested/", folder_name_destino)
        #print('Moviendo blob: ', destino_blob)
        bucket.rename_blob(blob, destino_blob)

while True:
 # try:
   bucket = storage_client.get_bucket(bucket_name)
   blobs = bucket.list_blobs(prefix = 'tested/');
   if blobs:
     for blob in blobs:
       relocate_blobs(blob)
       #time.sleep(1)
   else:
       print('No hay audios para reubicar.')
 # except:
  #    print('Error while relocating audios.')

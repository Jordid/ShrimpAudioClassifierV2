import requests

url = "http://35.198.46.91:80/proccessAudio"

files_to_be_uploaded = {
  "file": open("assets/audios/test_09_11_2022 15_19_43 - 15_20_12.wav", "rb"),
}

response = requests.post(url=url, files=files_to_be_uploaded)

print(response)
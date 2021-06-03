import pandas as pd
import matplotlib.pyplot as plot
import librosa 
from pydub import AudioSegment
from tempfile import mktemp
import sklearn
import librosa.display
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import imagehash
import pylab
import xlwt
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("hashes",cell_overwrite_ok=True)
song_number=0
hash_sheet=0
sheet_Number=0
song_index=0
songs_names=[]
#hash lists
hash_spectral_centroid_feature=[]
hash_spectral_rolloff_feature=[]
hash_mfcc_feature=[]
hash_chroma_stft=[]
 

features=['spectral_centroid','spectral_rolloff','Mel-Frequency Cepstral','Chroma']

for songName in os.listdir():
    if songName.endswith(".mp3"):
       songs_names.append(songName)
      

for songName in songs_names:
       mp3_audio = AudioSegment.from_file(songName, format="mp3")[:60000]  # read mp3
       wave_name= mktemp('.wav')  # use temporary file
       mp3_audio.export(wave_name, format="wav")  # convert to wav
       song_wave,samplingFrequency =librosa.load(wave_name,duration=60) 
       song_spectro= librosa.amplitude_to_db(np.abs(librosa.stft(song_wave)), ref=np.max)

       spectral_centroid_feature= librosa.feature.spectral_centroid(y=song_wave, sr=samplingFrequency)
       hash_spectral_centroid_feature.append(str((imagehash.phash(Image.fromarray( spectral_centroid_feature)))))

       spectral_rolloff_feature= librosa.feature.spectral_rolloff(y=song_wave, sr=samplingFrequency)
       hash_spectral_rolloff_feature.append(str((imagehash.phash(Image.fromarray(spectral_rolloff_feature)))))

       mfcc_feature = librosa.feature.mfcc(song_wave, samplingFrequency) #Mel-Frequency Cepstral Coefficients(MFCCs)
       hash_mfcc_feature.append(str((imagehash.phash(Image.fromarray(mfcc_feature)))))

       chroma_stft = librosa.feature.chroma_stft(song_wave, samplingFrequency) #Chroma feature
       hash_chroma_stft.append(str((imagehash.phash(Image.fromarray(chroma_stft)))))

df = pd.DataFrame({'SongsNames':songs_names,
    'Centroid_feature_hash':hash_spectral_centroid_feature,
    'Rolloff_feature_hash':hash_spectral_rolloff_feature,
    'Mfcc_feature_hash':hash_mfcc_feature,
    'Chroma_stft':hash_chroma_stft})

df.to_excel('./DataBase.xlsx', sheet_name='Hashes', index=False)

      

       


       
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

""" # you need to be in the folder of songs
""The spectral centroid indicates at which frequency the energy of a spectrum is centered upon or in other words It indicates where the ” center of mass” for a sound is located.
""A chroma feature or vector is typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, …, B}, is present in the signal. In short, It provides a robust way to describe a similarity measure between music pieces.
""A Spectral Rolloff is a measure of the shape of the signal. It represents the frequency at which high frequencies decline to 0. To obtain it, we have to calculate the fraction of bins in the power spectrum where 85% of its power is at lower frequencies.
""The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.
    """    
#directory="E:/songs" 
features=['spectral_centroid','spectral_rolloff','Mel-Frequency Cepstral','Chroma']
for songName in os.listdir():
    if songName.endswith(".mp3"):
       songs_names.append(songName)
for feature in features:
        song_number+=1
        sheet.write(0,song_number,feature)
for songname in songs_names:
       song_index+=1
       sheet.write(song_index,0,songname)
for songName in songs_names:
       mp3_audio = AudioSegment.from_file(songName, format="mp3")[:60000]  # read mp3
       wave_name= mktemp('.wav')  # use temporary file
       mp3_audio.export(wave_name, format="wav")  # convert to wav
       # Read the wav file (mono)
       song_wave,samplingFrequency =librosa.load(wave_name,duration=60) 
       song_spectro= librosa.amplitude_to_db(np.abs(librosa.stft(song_wave)), ref=np.max)
       spectral_centroid_feature= librosa.feature.spectral_centroid(y=song_wave, sr=samplingFrequency)
       hash_spectral_centroid_feature=str((imagehash.phash(Image.fromarray( spectral_centroid_feature))))
       spectral_rolloff_feature= librosa.feature.spectral_rolloff(y=song_wave, sr=samplingFrequency)
       hash_spectral_rolloff_feature=str((imagehash.phash(Image.fromarray(spectral_rolloff_feature))))
       mfcc_feature = librosa.feature.mfcc(song_wave, samplingFrequency) #Mel-Frequency Cepstral Coefficients(MFCCs)
       hash_mfcc_feature=str((imagehash.phash(Image.fromarray(mfcc_feature))))
       chroma_stft = librosa.feature.chroma_stft(song_wave, samplingFrequency) #Chroma feature
       hash_chroma_stft=str((imagehash.phash(Image.fromarray(chroma_stft))))
       hashes=[hash_mfcc_feature,hash_chroma_stft,hash_spectral_centroid_feature,hash_spectral_rolloff_feature]
       for Hash in hashes:
           hash_sheet+=1
           sheet.write(sheet_Number, hash_sheet,Hash)
       hash_sheet=0
       sheet_Number+=1
       hashes.clear()
workbook.save("DataBase.xls")

       


       
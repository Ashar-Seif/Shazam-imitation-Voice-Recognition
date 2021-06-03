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

""" # you need to be in the folder of songs
""The spectral centroid indicates at which frequency the energy of a spectrum is centered upon or in other words It indicates where the ” center of mass” for a sound is located.
""A chroma feature or vector is typically a 12-element feature vector indicating how much energy of each pitch class, {C, C#, D, D#, E, …, B}, is present in the signal. In short, It provides a robust way to describe a similarity measure between music pieces.
""A Spectral Rolloff is a measure of the shape of the signal. It represents the frequency at which high frequencies decline to 0. To obtain it, we have to calculate the fraction of bins in the power spectrum where 85% of its power is at lower frequencies.
""The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.

    """    
#directory="E:/songs" 
features=['spectral_centroid','spectral_rolloff','Mel-Frequency Cepstral','Chroma']
for songName in os.listdir():
#songName = 'Adele_Million_Years_Ago_10.mp3'
    if songName.endswith(".mp3"):
       mp3_audio = AudioSegment.from_file(songName, format="mp3")[:60000]  # read mp3
       wave_name= mktemp('.wav')  # use temporary file
       mp3_audio.export(wave_name, format="wav")  # convert to wav
       # Read the wav file (mono)
       song_wave,samplingFrequency =librosa.load(wave_name,duration=60) 
       Spectro_Path = 'spectros/'+os.path.splitext(os.path.basename(songName ))[0]+'.png'
       pylab.axis('off')  # no axis
       pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
       wave_abs= librosa.amplitude_to_db(np.abs(librosa.stft(song_wave)), ref=np.max)
       librosa.display.specshow(wave_abs, y_axis='linear')
       pylab.savefig(Spectro_Path, bbox_inches=None, pad_inches=0)
       SavePath = 'hashes_centroid/'+os.path.splitext(os.path.basename(songName))[0]+'Hash_centroid.png'
       spectral_centroid_feature= librosa.feature.spectral_centroid(y=song_wave, sr=samplingFrequency)
       librosa.display.specshow(spectral_centroid_feature.T,sr=samplingFrequency )
       pylab.savefig(SavePath, bbox_inches=None, pad_inches=0)
       SavePath = 'hashes_rolloff/'+os.path.splitext(os.path.basename(songName))[0]+'Hash_rolloff.png'
       spectral_rolloff_feature= librosa.feature.spectral_rolloff(y=song_wave, sr=samplingFrequency)
       librosa.display.specshow(spectral_rolloff_feature.T,sr=samplingFrequency )
       pylab.savefig(SavePath, bbox_inches=None, pad_inches=0)
       SavePath = 'hashes_mfcc/'+os.path.splitext(os.path.basename(songName))[0]+'hashes_mfcc.png'
       mfcc_feature = librosa.feature.mfcc(song_wave, samplingFrequency) #Mel-Frequency Cepstral Coefficients(MFCCs)
       librosa.display.specshow(mfcc_feature.T,sr=samplingFrequency )
       pylab.savefig(SavePath, bbox_inches=None, pad_inches=0)
       SavePath = 'hashes_chroma/'+os.path.splitext(os.path.basename(songName))[0]+'hashes_mfcc.png'
       chroma_stft = librosa.feature.chroma_stft(song_wave, samplingFrequency) #Chroma feature
       librosa.display.specshow(chroma_stft.T,sr=samplingFrequency )
       pylab.savefig(SavePath, bbox_inches=None, pad_inches=0)
       pylab.close()
       


       
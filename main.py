from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog , QPushButton,QMessageBox,QTableWidgetItem
from PyQt5 import QtWidgets, QtCore,QtGui,uic,QtMultimedia
from mainwindow import Ui_MainWindow
from scipy.spatial import distance
import matplotlib.pyplot as plot
from pydub import AudioSegment
from tempfile import mktemp
import os
import sys
import librosa 
#import sklearn
import librosa.display
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import pylab
import logging
import itertools 

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(lineno)s - %(levelname)s - %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
    #======================================================= UI Actions ===================================================================
        #Load the UI Page
        self.MainWindow = MainWindow
        uic.loadUi('mainwindow.ui', self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self) 
        self.LoadButtons= [self.ui.open_mix1 , self.ui.open_mix2]
        for i in range(len(self.LoadButtons)):
            self.Load_Buttons_Counter(i)
        self.ui.Mixer_songs.setMaximum(100)
        self.ui.Mixer_songs.setMinimum(0)
        self.ui.Mixer_songs.setSingleStep(1)
        self.ui.Mixer_songs.setValue(0)
        self.ui.Mixer_songs.valueChanged.connect(lambda : self.mix_songs())
        self.ui.show_results.clicked.connect(lambda : self.Data_Base_Comparing()) 
        self.ui.actionsong.triggered.connect(lambda : self.open_single_song())
        self.songs_waves=[...,...,...]
        self.mfcc_feature =[...,...]
        self.strings=[...,...]
        #Comparison_lists
        self.DB_songs_names=[]
        self.DB_Centroid_feature_hash=[]
        self.DB_Rolloff_feature_hash=[]
        self.DB_Mfcc_feature_hash=[]
        self.DB_Chroma_stft_feature_hash=[]
        logger.info("The Application started successfully")
        df = pd.read_excel ('Features.xlsx')
        DB_hashes=[]
        for i, row in df.iterrows():
            self.DB_songs_names.append(f"{row['SongsNames']}")
            self.DB_Centroid_feature_hash.append(list(f"{row['Centroid_feature_hash']}"))
            self.DB_Rolloff_feature_hash.append(list(f"{row['Rolloff_feature_hash']}"))
            self.DB_Mfcc_feature_hash.append(list(f"{row['Mfcc_feature_hash']}"))
            self.DB_Chroma_stft_feature_hash.append(list(f"{row['Chroma_stft']}"))
            self.DB_hashes=[self.DB_Centroid_feature_hash,self.DB_Rolloff_feature_hash,self.DB_Mfcc_feature_hash,self.DB_Chroma_stft_feature_hash]
 #============================================== Functions of loading  and reading Songs ==============================================
    def mp3_To_wav_Converter(self,songID):
      options =  QtWidgets.QFileDialog.Options()
      fname = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
                        "*.mp3", options=options)
      self.path = fname[0] 
      print(self.path)
      if self.path =="" :
          pass
      mp3_audio = AudioSegment.from_file( self.path , format="mp3")[:60000]  # read mp3
      wave_name = mktemp('.wav')  # use temporary file
      mp3_audio.export(wave_name, format="wav")  # convert to wav
      self.songs_waves[songID],self.samplingFrequency =librosa.load(wave_name,duration=60) 
      print(self.samplingFrequency)
      print(f"Song {songID+1} has been Uploaded")
      logger.info(f"Song{songID+1} has been Uploaded")
      if(all(type(song) != type(...) for song in  self.songs_waves)):
         self.mix_songs()
      else:
        self.Single_Song=self.songs_waves[songID]
        self.song_features(self.Single_Song)
    def open_single_song(self):
      self.mp3_To_wav_Converter(0)
      self.song_features(self.songs_waves[0])

#============================================== Function Mixing Songs ==============================================
    def mix_songs(self) :
      songs_ratio = self.ui.Mixer_songs.value()/100
      self.Mixed_Song = self.songs_waves[0] * songs_ratio + self.songs_waves[1] * (1-songs_ratio)
      self.ui.percentagemix.setText(str(songs_ratio*100)+"%")
      logger.info(f"Songs has been mixed correctly")
      self.song_features(self.Mixed_Song)
      self.Data_Base_Comparing()
#============================================== Function of Extracting Songs feature ==============================================
    def song_features (self,Song):
        """spectrogram [Makes the spectrogram of a song , calls a function to hash the the spectro. and a function to extract the features of the song]
        """
        self.features_hashes=[]
        #Song spectogram
        song_spectogram= librosa.amplitude_to_db(np.abs(librosa.stft(Song)), ref=np.max)
        spectral_centroid_feature= librosa.feature.spectral_centroid(Song, self.samplingFrequency)
        spectral_rolloff_feature= librosa.feature.spectral_rolloff(Song, self.samplingFrequency)
        mfcc_feature = librosa.feature.mfcc(Song, self.samplingFrequency)#Mel-Frequency Cepstral Coefficients(MFCCs)
        chroma_stft_feature = librosa.feature.chroma_stft(Song, self.samplingFrequency) #Chroma feature
        features=[spectral_centroid_feature,spectral_rolloff_feature, mfcc_feature,chroma_stft_feature]
        for feature in features:
          self.features_hashes.append(list(str((imagehash.phash(Image.fromarray(feature))))))
#============================================== Function of comparing song with database  ==============================================       
    def Data_Base_Comparing(self):
        self.Distancing_Hamming_reverse=[[],[],[],[]]
        self.similarity_index=[]
        for i in range(len(self.DB_Centroid_feature_hash)): 
          for j in range(len(self.features_hashes)):
              self.Distancing_Hamming_reverse[j].append(1-distance.hamming(self.DB_hashes[j][i], self.features_hashes[j]))
          self.similarity_index.append(100*((self.Distancing_Hamming_reverse[0][i]+self.Distancing_Hamming_reverse[1][i]+0.5*self.Distancing_Hamming_reverse[2][i]+0.5*self.Distancing_Hamming_reverse[3][i])/3))  
        self.Sorting_songs(self.similarity_index)
  #============================================== Function of sorting the closest Songs ==============================================
    def Sorting_songs(self,similarity_list):
      sort_songs=self.DB_songs_names.copy()
      self.similarity=[]
      self.Songs_list=[]
      for maximum in range(10):
        max_similarity = 0
        for song_similarity in range(len(similarity_list)):
          if similarity_list[song_similarity] >  max_similarity:
            max_similarity = similarity_list[song_similarity]
        max_index=similarity_list.index(max_similarity)
        #List of songs to be displayed in table 
        self.Songs_list.append(sort_songs[max_index])
        similarity_list.pop(max_index)
        sort_songs.pop(max_index)
        #List of similarity_index to be displayed in table
        self.similarity.append(str(max_similarity))
        newItem = QtWidgets.QTableWidgetItem(self.Songs_list[maximum ])
        self.ui.similarity_output_table.setItem(maximum-1,2,newItem)
        newItem2 = QtWidgets.QTableWidgetItem(self.similarity[maximum])
        self.ui.similarity_output_table.setItem(maximum-1,3,newItem2)
  
    def Load_Buttons_Counter(self,i:int):
          self.LoadButtons[i].clicked.connect(lambda : self.mp3_To_wav_Converter(i))     
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':      
 main()

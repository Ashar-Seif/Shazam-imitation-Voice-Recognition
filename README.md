# Shazam-imitation-Voice-Recognition
 A basic implementation of shazam like  audio Recognition algorithm, it utilizes the advantages of a spectrogram and perceptual hashing :</n>

1- A database is formed of 241 songs (mp3 audio File) separated to their Vocals and music.

2- Extraction of Spectrogram and spectral Features (Spectral centroid,Spectral Rolloff,Mel frequency Coefficient and Chroma STFT) is executed.

3- Hashing the extracted data with a Perceptual Hashing Algorithm.

4- A test Song is given to the application with extraction of its Hash the matches are found.

5- Matching percentages are calculated according to a distance Hamming function.

6- A testing mechanism is implemented by mixing two Audio files then this mix is given to the application to find it's matches in the database.


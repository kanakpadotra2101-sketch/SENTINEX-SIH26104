import librosa
import numpy as np 
def preprocess_audio( file_path ) : 
  # Load audio 16kHz mono
  y , sr = librosa.load( file_path , sr=16000 ) 
# VAD : trim silence 
y_trimmed , _ = librosa.effects.trim(y , top_db=25 ) 
# MFCC features for model 
mfcc = librosa.feature.mfcc( y=y_trimmed , sr=sr , n_mfcc=40 )
return mfcc , y_trimmed , sr 

import librosa
import numpy as np 
def preprocess_audio(file_path) : 
  y , sr = librosa.load(file_path,
                         sr=16000) 
    # Trim silence 
  y_trimmed , _ = librosa.effects.trim(y , top_db=25) 
    # Extract MFCC features
  mfcc = librosa.feature.mfcc ( 
    y=y_trimmed ,
    sr=sr ,
      n_mfcc=40
        )
  return      mfcc , y_trimmed , sr
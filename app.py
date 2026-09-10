from fastapi import FastAPI , UploadFile 
from fastapi.staticfiles import StaticFiles 
from preprocessing import preprocess_audio 
from model import VoiceCloneModel 
import tempfile
import os
app = FastAPI( title="Voice Shield AI" )
model = VoiceCloneModel()
@app.post("/predict") 
async def predict (file: UploadFile) :
  audio_data = await file.read() 
  with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
    temp.write(audio_data)
    temp_path = temp.name
  try:
    mfcc , audio , sr = preprocess_audio(temp_path) 
    result = model.predict(mfcc) 
    return {"result": result}
  finally:
    os.remove(temp_path)
app.mount("/" , StaticFiles(directory="static" , html=True), name ="static")
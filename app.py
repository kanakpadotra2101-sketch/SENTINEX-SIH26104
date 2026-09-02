from fastapi import FastAPI , UploadFile 
from fastapi.staticfiles import StaticFiles 
from preprocessing import preprocess_audio 
from model import VoiceShieldModel 
app = FastAPI( title="SENTINEX - Voice Shield AI" )
model = VoiceShieldModel()
@app.post("/predict") 
async def predict (file: UploadFile) : 
  temp_path = f"temp_{file.filename}" 
with open ( temp_path , "wb" )  as f : 
  f.write(await file.read()) 
mfcc , audio , sr = preprocess_audio(temp_parh) 
result = model.predict(mfcc) 
return result 
app.mount("/" , StaticFiles(directory="static" , html=True) 

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from model import VoiceCloneModel

import librosa
import numpy as np
import tempfile
import subprocess
import os
import time


app = FastAPI(title="SENTINEX - Voice Shield AI")

# Load the AASIST model once when the server starts
model = VoiceCloneModel()

# AASIST expects 64600 audio samples at 16 kHz
TARGET_LENGTH = 64600


@app.post("/predict")
async def predict(file: UploadFile):

    start_time = time.perf_counter()

    audio_data = await file.read()
    upload_time = time.perf_counter() - start_time

    # Detect uploaded file type
    original_ext = os.path.splitext(file.filename or "")[1].lower()

    if not original_ext:
        original_ext = ".webm"

    input_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=original_ext
    ).name

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ).name

    try:

        # --------------------------------
        # SAVE UPLOADED AUDIO
        # --------------------------------

        with open(input_path, "wb") as f:
            f.write(audio_data)


        # --------------------------------
        # CONVERT AUDIO TO 16 kHz MONO WAV
        # --------------------------------

        convert_start = time.perf_counter()

        if original_ext == ".wav":

            output_path = input_path
            convert_time = 0.0

        else:

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    output_path
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            convert_time = time.perf_counter() - convert_start


        # --------------------------------
        # LOAD AUDIO
        # --------------------------------

        audio, sr = librosa.load(
            output_path,
            sr=16000,
            mono=True
        )

        if len(audio) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio contains no usable sound."
            )


        # --------------------------------
        # PREPARE AUDIO FOR AASIST
        # --------------------------------

        if len(audio) > TARGET_LENGTH:

            audio = audio[:TARGET_LENGTH]

        elif len(audio) < TARGET_LENGTH:

            audio = np.tile(
                audio,
                int(np.ceil(TARGET_LENGTH / len(audio)))
            )[:TARGET_LENGTH]


        # --------------------------------
        # RUN MODEL ONCE
        # --------------------------------

        model_start = time.perf_counter()

        result = model.predict(audio)

        model_time = time.perf_counter() - model_start


        # --------------------------------
        # TOTAL TIME
        # --------------------------------

        total_time = time.perf_counter() - start_time


        print("RESULT:", result)
        print(f"TOTAL ANALYSIS TIME: {total_time:.2f} seconds")
        print(f"CONVERT TIME: {convert_time:.2f} seconds")
        print(f"UPLOAD TIME: {upload_time:.2f} seconds")
        print(f"MODEL TIME: {model_time:.2f} seconds")


        # --------------------------------
        # SEND RESULT TO FRONTEND
        # --------------------------------

        return {
            "label": result["label"],
            "confidence": result["confidence"],
            "processing_time": round(total_time, 2)
        }


    except HTTPException:
        raise


    except subprocess.CalledProcessError:

        raise HTTPException(
            status_code=500,
            detail="Could not convert the uploaded audio."
        )


    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        # Remove temporary input file
        if os.path.exists(input_path):
            os.remove(input_path)

        # Remove temporary WAV file
        if (
            output_path != input_path
            and os.path.exists(output_path)
        ):
            os.remove(output_path)


# --------------------------------
# SERVE FRONTEND
# --------------------------------

app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)
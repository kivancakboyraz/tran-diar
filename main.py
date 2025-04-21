# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.pipeline import transcribe_with_diarization
import shutil
import os
import uuid

app = FastAPI()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.mp3"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = transcribe_with_diarization(filename)
        return JSONResponse(content={"segments": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        os.remove(filename)

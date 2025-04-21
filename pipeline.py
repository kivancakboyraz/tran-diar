from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import numpy as np
import gc

# Gerekli modelleri yükle
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
)

model = whisper.load_model("large-v2")

   
def read(audio_segment):
    """Pydub segmentini numpy array'e çevir"""
    y = np.array(audio_segment.get_array_of_samples())
    return np.float32(y) / 32768


def millisec(time_str):
    """Zaman stringini milisaniyeye çevir"""
    h, m, s = time_str.split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def transcribe_with_diarization(mp3_path: str):
    """Konuşmacı ayrımı + transkripsiyon"""
    diarization_result = str(pipeline(mp3_path)).splitlines()

    audio = AudioSegment.from_mp3(mp3_path).set_frame_rate(16000)

    results = []

    for line in diarization_result:
        parts = line.split(" ")
        if len(parts) < 7:
            continue  # Eksik veri varsa geç

        start = millisec(parts[1])
        end = millisec(parts[4][:-1])  # ']' karakteri silinir
        speaker = parts[6]

        segment = audio[start:end]
        samples = read(segment)

        transcription = model.transcribe(samples, fp16=True)

        results.append({
            "start": parts[1],
            "end": parts[3],
            "speaker": speaker,
            "text": transcription["text"]
        })

        # Temizlik
        del transcription, samples, segment
        gc.collect()

    return results

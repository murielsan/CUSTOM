from fastapi import APIRouter, UploadFile
from pydub import AudioSegment

# Population endpoint
router = APIRouter()

AUDIO_LENGTH = 10000

# Get stations list
@router.post("/predict/")
async def predict(audio: UploadFile):
    sound = AudioSegment.from_file(audio.file, format="wav")
    sound[:AUDIO_LENGTH]

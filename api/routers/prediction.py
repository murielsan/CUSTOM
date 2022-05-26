from fastapi import APIRouter, UploadFile
from pydub import AudioSegment
from utils.communications import resultsQueue
from utils.extract_features import FeatureExtractor
from utils.predictor import Predictor
from utils.sound_utils import SoundHelper

# Population endpoint
router = APIRouter()

AUDIO_LENGTH = 10000
AUDIO_TYPE = "stream"
TIMEOUT = 5

feature_extractor = FeatureExtractor(audio_type=AUDIO_TYPE)
feature_extractor.start()
predictor = Predictor()
predictor.start()
sound_helper = SoundHelper(AUDIO_TYPE)

# Get stations list
@router.post("/predict/")
async def predict(audio: UploadFile):
    """Receives an audiofile and predicts its class"""
    sound = AudioSegment.from_file(audio.file._file, codec="opus")
    sound += AudioSegment.silent(AUDIO_LENGTH - len(sound))  # Top up to 10 secs
    sound_helper.queue_sound(sound[:AUDIO_LENGTH])
    return resultsQueue.get()

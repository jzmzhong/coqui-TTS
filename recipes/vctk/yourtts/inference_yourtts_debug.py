import os
import torch
from TTS.api import TTS

MODEL_DIR = "/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS/accented_TTS/10_ckpts/01_baseline-January-03-2024_09+05AM-d6b3fb0"
CKPT = 30000
TEST_TXT_DIR = "/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS/accented_TTS/01_preprocessed/VCTK/txt/p234"
REFERENCE_WAV = "/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS/accented_TTS/01_preprocessed/VCTK/wav48_silence_trimmed/p234/p234_361_mic2.flac"
OUTPUT_WAV_FOLDER = os.path.join(MODEL_DIR, "checkpoint_{}".format(CKPT))

os.makedirs(OUTPUT_WAV_FOLDER, exist_ok=True)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS(
    model_path=os.path.join(MODEL_DIR, "checkpoint_{}.pth".format(CKPT)),
    config_path=os.path.join(MODEL_DIR, "config.json"),
    ).to(device)

# Run TTS
# Text to speech to a file
for TXT_FILE in os.listdir(TEST_TXT_DIR)[:30]:
    txt_path = os.path.join(TEST_TXT_DIR, TXT_FILE)
    with open(txt_path, encoding="utf-8", mode="r") as f:
        txt = f.read().strip()
    tts.tts_to_file(text=txt,
                    speaker="VCTK_p234",
                    # speaker_wav=REFERENCE_WAV,
                    # language="en",
                    file_path=os.path.join(OUTPUT_WAV_FOLDER, TXT_FILE.replace(".txt", ".wav")))

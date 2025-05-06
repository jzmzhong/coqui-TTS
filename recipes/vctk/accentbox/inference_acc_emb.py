import os
import torch
from TTS.api import TTS

DATA_ROOT = "/home/s2526235/AccentedTTS/data/VCTK-Corpus-0.92-24kHz"
MODEL_ROOT = "/home/s2526235/AccentedTTS/models"

MODEL2CKPTS = {
    "YourTTS-Finetune-VCTK-AccEmbV6-April-25-2025_06+32PM-97267563": [1400000],
}

TEST_TXT2SPK = {
        "p261": ["p261"],
        "p228": ["p228"],
        "p294": ["p294"],
        "p347": ["p347"],
        "p253": ["p253"],
        "p252": ["p252"],
        "p248": ["p248"],
        "p335": ["p335"],
        "p245": ["p245"],
        "p326": ["p326"],
        "p302": ["p302"],
    }

for MODEL_NAME, CKPTS in MODEL2CKPTS.items():
    MODEL_DIR = os.path.join(MODEL_ROOT, MODEL_NAME)
    for CKPT in CKPTS:

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Init TTS
        tts = TTS(
            model_path=os.path.join(MODEL_DIR, "checkpoint_{}.pth".format(CKPT)),
            config_path=os.path.join(MODEL_DIR, "config.json"),
            ).to(device)
        
        for TXT, SPKS in TEST_TXT2SPK.items():
            TEST_TXT_DIR = os.path.join(DATA_ROOT, "txt", TXT)
            for SPK in SPKS:
                
                REFERENCE_WAV = os.path.join(DATA_ROOT, "wav48_silence_trimmed", SPK, SPK+"_024_mic1.flac") 
                
                OUTPUT_WAV_FOLDER = os.path.join(MODEL_DIR, "wav", "ckpt-{}_refspk-{}_refuttr-024".format(str(CKPT)[:-3]+"k", SPK))
                os.makedirs(OUTPUT_WAV_FOLDER, exist_ok=True)

                # Run TTS
                # Text to speech to a file
                for TXT_FILE in sorted(os.listdir(TEST_TXT_DIR))[:23]:
                    txt_id = TXT_FILE.split("_")[-1].split(".")[0]
                    if int(txt_id) > 23:
                        continue
                    txt_path = os.path.join(TEST_TXT_DIR, TXT_FILE)
                    
                    with open(txt_path, encoding="utf-8", mode="r") as f:
                        txt = f.read().strip()
                    tts.tts_to_file(text=txt,
                                    speaker="VCTK_{}".format(SPK),
                                    speaker_wav=REFERENCE_WAV,
                                    accent_wav=REFERENCE_WAV,
                                    language="en",
                                    file_path=os.path.join(OUTPUT_WAV_FOLDER, TXT_FILE.replace(".txt", ".wav")))

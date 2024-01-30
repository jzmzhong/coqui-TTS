import os
import torch
from TTS.api import TTS

DATA_ROOT = "/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS/accented_TTS"

MODEL2CKPTS = {
    "01_baseline-January-13-2024_09+03PM-2016a95": [140000],
}

TEST_TXT2SPK = {
    "p225": ["p225", "p234", "p238", "p245", "p248", "p261", "p294", "p302", "p326", "p335", "p347"],
    "p234": ["p234"],
    "p238": ["p238"],
    "p245": ["p245"],
    "p248": ["p248"],
    "p261": ["p261"],
    "p294": ["p294"],
    "p302": ["p302"],
    "p326": ["p326"],
    "p335": ["p335"],
    "p347": ["p347"],
}

for MODEL_NAME, CKPTS in MODEL2CKPTS.items():
    MODEL_DIR = os.path.join(DATA_ROOT, "10_ckpts", MODEL_NAME)
    for CKPT in CKPTS:
        for TXT, SPKS in TEST_TXT2SPK.items():
            TEST_TXT_DIR = os.path.join(DATA_ROOT, "01_preprocessed/VCTK/txt/", TXT)
            for SPK in SPKS:
                REFERENCE_WAV = os.path.join(DATA_ROOT, "01_preprocessed/VCTK/wav48_silence_trimmed", SPK, SPK+"_001_mic1.flac") 
                OUTPUT_WAV_FOLDER = os.path.join(MODEL_DIR, "text_{}_speaker_{}_checkpoint_{}".format(TXT, SPK, CKPT))

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
                                    speaker="VCTK_{}".format(SPK),
                                    speaker_wav=REFERENCE_WAV,
                                    language="en",
                                    file_path=os.path.join(OUTPUT_WAV_FOLDER, TXT_FILE.replace(".txt", ".wav")))

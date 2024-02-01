import os
import torch
from TTS.api import TTS

DATA_ROOT = "/Users/jeffzhong/Desktop/work/Transsion/04_AccentedTTS/accented_TTS"

MODEL2CKPTS = {
    # "10_acc-clf-enc-January-13-2024_10+38PM-2016a95": [140000],
    "11_acc-alpha100-January-30-2024_04+42PM-6801829": [135000],
    "12_acc-alpha1000-January-30-2024_04+44PM-6801829": [135000],
    "13_acc-alpha10000-January-30-2024_04+47PM-6801829": [135000],
}

TEST_TXT2SPK = {
    "p225": [("p225", "English"),
             ("p225", "American"),
             ("p225", "Irish"),
             ("p225", "Scottish"),
             ("p234", "Scottish"),
             ("p245", "Irish"),
             ("p294", "American"),
    ],
    "p234": [("p234", "Scottish"),],
    "p245": [("p245", "Irish")],
    "p294": [("p294", "American")],
}

for MODEL_NAME, CKPTS in MODEL2CKPTS.items():
    MODEL_DIR = os.path.join(DATA_ROOT, "10_ckpts", MODEL_NAME)
    for CKPT in CKPTS:
        for TXT, SPKS in TEST_TXT2SPK.items():
            TEST_TXT_DIR = os.path.join(DATA_ROOT, "01_preprocessed/VCTK/txt/", TXT)
            for (SPK, ACC) in SPKS:
                REFERENCE_WAV = os.path.join(DATA_ROOT, "01_preprocessed/VCTK/wav48_silence_trimmed", SPK, SPK+"_002_mic1.flac") 
                # OUTPUT_WAV_FOLDER = os.path.join(MODEL_DIR, "text_{}_speaker_{}_accent_{}_intensity_ref_checkpoint_{}".format(TXT, SPK, ACC, CKPT))
                OUTPUT_WAV_FOLDER = os.path.join(MODEL_DIR, "debug_text_{}_speaker_{}_accent_{}_intensity_ref_checkpoint_{}".format(TXT, SPK, ACC, CKPT))

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
                                    language=ACC,
                                    file_path=os.path.join(OUTPUT_WAV_FOLDER, TXT_FILE.replace(".txt", ".wav")))

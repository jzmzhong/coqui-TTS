import os
import argparse
import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""

parser = argparse.ArgumentParser()

# basic exp, data, model set-up
parser.add_argument(
    "--exp-name",
    type=str,
    default="",
    required=True,
    help="Experiment name for storing logs and checkpoints.",
)
parser.add_argument(
    "--preprocessed-dataset-path",
    type=str,
    default="",
    required=True,
    help="The path to the preprocessed dataset.",
)
parser.add_argument(
    "--out-path",
    type=str,
    default="",
    required=True,
    help="The path to the saved checkpoints.",
)

# training set-up
parser.add_argument(
    "--restore-path",
    type=str,
    default=None,
    required=False,
    help="The path to the restored checkpoint.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    required=False,
    help="The batch size for training, default set as 32.",
)

# audio set-up
parser.add_argument(
    "--raw-dataset-path",
    type=str,
    default=None,
    required=False,
    help="The path to the raw dataset.",
)
parser.add_argument(
    "--sample-rate",
    type=int,
    default=16000,
    required=False,
    help="The sample rate for audios, default set as 16000.",
)


args = parser.parse_args()

EXP_NAME = args.exp_name
PREPROCESSED_DATASET_PATH = args.preprocessed_dataset_path
OUT_PATH = args.out_path

RESTORE_PATH = args.restore_path
BATCH_SIZE = args.batch_size
# This paramter is useful to debug, it skips the training epochs and just do the evaluation and produce the test sentences
SKIP_TRAIN_EPOCH = False

RAW_DATASET_PATH = args.raw_dataset_path
SAMPLE_RATE = args.sample_rate
# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 10
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10

# audio download and preprocess
if RAW_DATASET_PATH:
    if not os.path.exists(RAW_DATASET_PATH):
        print(">>> Downloading VCTK dataset:")
        download_vctk(RAW_DATASET_PATH)
    resample_files(RAW_DATASET_PATH, SAMPLE_RATE, output_dir=PREPROCESSED_DATASET_PATH, file_ext="flac", n_jobs=NUM_RESAMPLE_THREADS)

# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    meta_file_train="",
    meta_file_val="",
    path=PREPROCESSED_DATASET_PATH,
    language="en",
    ignored_speakers=[
        "p261",
        "p225",
        "p294",
        "p347",
        "p238",
        "p234",
        "p248",
        "p335",
        "p245",
        "p326",
        "p302",
    ],  # Ignore the test speakers to full replicate the paper experiment
)

# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to add new datasets, just add them here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [vctk_config]

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)



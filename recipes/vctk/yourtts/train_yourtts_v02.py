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
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager

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
parser.add_argument(
    "--lang-emb-dim",
    type=int,
    default=0,
    required=False,
    help="The dimension for language embedding, default set as 0 (not using one).",
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
LANG_EMB_DIM = args.lang_emb_dim

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
DATASETS_CONFIG_LIST = []
# for LANG in ["American", "Australian", "Canadian", "English", "Indian", "Irish", "NewZealand", "NorthernIrish", "Scottish", "Welsh"]:
# for LANG in ["American", "Australian", "Canadian", "English", "Indian", "Irish", "NorthernIrish", "Scottish"]: # remove single speaker accent dataset
for LANG in ["American", "English", "Irish", "Scottish"]: # remove few speakers accent dataset
    DATASETS_CONFIG_LIST.append(BaseDatasetConfig(
        formatter="vctk",
        dataset_name="vctk",
        meta_file_train="",
        meta_file_val="",
        path=os.path.join(PREPROCESSED_DATASET_PATH, LANG),
        language=LANG,
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
    ))

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


# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    # use_speaker_encoder_as_loss=True,
    # Useful parameters to enable multilingual training
    use_language_embedding=(LANG_EMB_DIM!=0),
    embedded_language_dim=LANG_EMB_DIM,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=EXP_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=0,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=50,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        # characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? ",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? ",
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "VCTK_p277",
            None,
            "English",
        ],
        [
            "Be a voice, not an echo.",
            "VCTK_p239",
            None,
            "English",
        ],
        [
            "I'm sorry Dave. I'm afraid I can't do that.",
            "VCTK_p258",
            None,
            "English",
        ],
        [
            "This cake is great. It's so delicious and moist.",
            "VCTK_p244",
            None,
            "English",
        ],
        [
            "Prior to November 22, 1963.",
            "VCTK_p305",
            None,
            "American",
        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model

# init language manager for multi-accent training
language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

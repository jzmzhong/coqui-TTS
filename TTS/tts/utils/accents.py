import json
import os
from typing import Any, Dict, List, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default
from TTS.tts.utils.managers import EmbeddingManager


class AccentManager(EmbeddingManager):
    """Manage the accents for multi-accent multi-speaker 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by accent or clip.

    There are 3 different scenarios considered:

    1. Models using accent embedding layers. The datafile only maps speaker names to accent ids used by the embedding layer.
    2. Models using d-vectors (accent). The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'speakerA',
                'embedding'[<d_vector_accent_values>]
            },
            ...
        }


    3. Computing the d-vectors (accent) by the accent encoder. It loads the accent encoder model and
    computes the d-vectors (accent) for a given accent or clip.

    Args:
        d_vectors_accent_file_path (str, optional): Path to the metafile including d-vectors (accent). Defaults to "".
        accent_id_file_path (str, optional): Path to the metafile that maps speaker names to accent ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the accent encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the accent encoder config file. Defaults to "".

    Examples:
        >>> # load audio processor and accent encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = AccentManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> waveform = ap.load_wav(sample_wav_path)
        >>> mel = ap.melspectrogram(waveform)
        >>> d_vector = manager.compute_embeddings(mel.T)
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        d_vectors_accent_file_path: str = "",
        accent_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):
        super().__init__(
            embedding_file_path=d_vectors_accent_file_path,
            id_file_path=accent_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="speaker_name")

    @property
    def num_accents(self):
        return len(self.name_to_id)

    @property
    def accent_names(self):
        return list(self.name_to_id.keys())

    def get_accents(self) -> List:
        return self.name_to_id

    @staticmethod
    def init_from_config(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "AccentManager":
        """Initialize an accent manager from config

        Args:
            config (Coqpit): Config object.
            samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the speaker names.
                Defaults to None.

        Returns:
            AccentEncoder: Accent encoder object.
        """
        accent_manager = None
        if get_from_config_or_model_args_with_default(config, "use_accent_embedding", False):
            if samples:
                accent_manager = AccentManager(data_items=samples)
            if get_from_config_or_model_args_with_default(config, "accent_file", None):
                accent_manager = AccentManager(
                    speaker_id_file_path=get_from_config_or_model_args_with_default(config, "accent_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "accents_file", None):
                accent_manager = AccentManager(
                    speaker_id_file_path=get_from_config_or_model_args_with_default(config, "accents_file", None)
                )

        if get_from_config_or_model_args_with_default(config, "use_d_vector_accent_file", False):
            accent_manager = AccentManager()
            if get_from_config_or_model_args_with_default(config, "d_vector_accent_file", None):
                accent_manager = AccentManager(
                    d_vectors_accent_file_path=get_from_config_or_model_args_with_default(config, "d_vector_accent_file", None)
                )
        return accent_manager
    
    # overwrite inherited function
    def init_encoder(self, model_path: str, config_path: str, use_cuda=False) -> None:
        """Initialize a speaker encoder model.

        Args:
            model_path (str): Model file path.
            config_path (str): Model config file path.
            use_cuda (bool, optional): Use CUDA. Defaults to False.
        """
        self.use_cuda = use_cuda
        self.encoder_config = load_config(config_path)
        self.encoder = setup_encoder_model(self.encoder_config)
        self.encoder_criterion = self.encoder.load_checkpoint(
            self.encoder_config, model_path, eval=True, use_cuda=use_cuda, cache=True
        )
        self.encoder_ap = AudioProcessor(**self.encoder_config.audio)

    def compute_embedding_from_clip(self, wav_file: Union[str, List[str]]) -> list:
        """Compute a embedding from a given audio file.

        Args:
            wav_file (Union[str, List[str]]): Target file path.

        Returns:
            list: Computed embedding.
        """

        def _compute(wav_file: str):
            waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
            if not self.encoder_config.model_params.get("use_torch_spec", False):
                m_input = self.encoder_ap.melspectrogram(waveform)
                m_input = torch.from_numpy(m_input)
            else:
                m_input = torch.from_numpy(waveform)

            if self.use_cuda:
                m_input = m_input.cuda()
            m_input = m_input.unsqueeze(0)
            embedding = self.encoder.compute_embedding(m_input)
            return embedding

        if isinstance(wav_file, list):
            # compute the mean embedding
            embeddings = None
            for wf in wav_file:
                embedding = _compute(wf)
                if embeddings is None:
                    embeddings = embedding
                else:
                    embeddings += embedding
            return (embeddings / len(wav_file))[0].tolist()
        embedding = _compute(wav_file)
        return embedding[0].tolist()

    def compute_embeddings(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
        """Compute embedding from features.

        Args:
            feats (Union[torch.Tensor, np.ndarray]): Input features.

        Returns:
            List: computed embedding.
        """
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if self.use_cuda:
            feats = feats.cuda()
        return self.encoder.compute_embedding(feats)


def _set_file_path(path):
    """Find the accents.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "accents.json")
    path_continue = os.path.join(path, "accents.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `accents.json` not found in {path}")


def load_accent_mapping(out_path):
    """Loads accent mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with fsspec.open(json_file, "r") as f:
        return json.load(f)


def save_accent_mapping(out_path, accent_mapping):
    """Saves accent mapping if not yet present."""
    if out_path is not None:
        accents_json_path = _set_file_path(out_path)
        with fsspec.open(accents_json_path, "w") as f:
            json.dump(accent_mapping, f, indent=4)


def get_accent_manager(c: Coqpit, data: List = None, restore_path: str = None, out_path: str = None) -> AccentManager:
    """Initiate a `AccentManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        data (List): Data samples used in training to infer speakers from. It must be provided if speaker embedding
            layers is used. Defaults to None.
        out_path (str, optional): Save the generated speaker IDs to a output path. Defaults to None.

    Returns:
        AccentManager: initialized and ready to use instance.
    """
    accent_manager = AccentManager()
    if c.use_accent_embedding:
        if data is not None:
            accent_manager.set_ids_from_data(data, parse_key="speaker_name")
        if restore_path:
            accents_file = _set_file_path(restore_path)
            # restoring accent manager from a previous run.
            if c.use_d_vector_accent_file:
                # restore accent manager with the embedding file
                if not os.path.exists(accents_file):
                    print("WARNING: accents.json was not found in restore_path, trying to use CONFIG.d_vector_accent_file")
                    if not os.path.exists(c.d_vector_accent_file):
                        raise RuntimeError(
                            "You must copy the file accents.json to restore_path, or set a valid file in CONFIG.d_vector_accent_file"
                        )
                    accent_manager.load_embeddings_from_file(c.d_vector_accent_file)
                accent_manager.load_embeddings_from_file(accents_file)
            elif not c.use_d_vector_accent_file:  # restor accent manager with accent ID file.
                accent_ids_from_data = accent_manager.name_to_id
                accent_manager.load_ids_from_file(accents_file)
                assert all(
                    accent in accent_manager.name_to_id for accent in accent_ids_from_data
                ), " [!] You cannot introduce new accents to a pre-trained model."
        elif c.use_d_vector_accent_file and c.d_vector_accent_file:
            # new accent manager with external accent embeddings.
            accent_manager.load_embeddings_from_file(c.d_vector_accent_file)
        elif c.use_d_vector_accent_file and not c.d_vector_accent_file:
            raise "use_use_d_vector_accent_file is True, so you need pass a external accent embedding file."
        elif c.use_accent_embedding and "accents_file" in c and c.accents_file:
            # new accent manager with speaccentaker IDs file.
            accent_manager.load_ids_from_file(c.accents_file)

        if accent_manager.num_accents > 0:
            print(
                " > Accent manager is loaded with {} accents: {}".format(
                    accent_manager.num_accents, ", ".join(accent_manager.name_to_id)
                )
            )

        # save file if path is defined
        if out_path:
            out_file_path = os.path.join(out_path, "accents.json")
            print(f" > Saving `accents.json` to {out_file_path}.")
            if c.use_d_vector_accent_file and c.d_vector_accent_file:
                accent_manager.save_embeddings_to_file(out_file_path)
            else:
                accent_manager.save_ids_to_file(out_file_path)
    return accent_manager


def get_accent_balancer_weights(items: list):
    raise NotImplementedError
    # speaker_names = np.array([item["speaker_name"] for item in items])
    # unique_speaker_names = np.unique(speaker_names).tolist()
    # speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    # speaker_count = np.array([len(np.where(speaker_names == l)[0]) for l in unique_speaker_names])
    # weight_speaker = 1.0 / speaker_count
    # dataset_samples_weight = np.array([weight_speaker[l] for l in speaker_ids])
    # # normalize
    # dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    # return torch.from_numpy(dataset_samples_weight).float()

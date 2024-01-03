import os
from tqdm import tqdm
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

VCTK_accent_split_folder = "/data/zuomu/accented_TTS/01_preprocessed/VCTK_accent_split"
LANGs = ["American", "Australian", "Canadian", "English", "Indian", "Irish", "NewZealand", "NorthernIrish", "Scottish", "Welsh"]\

embs = []
embs_info = []

SPK_EMBs = {}
for LANG in LANGs:
    SPK2WAV = {}
    temp = torch.load(os.path.join(VCTK_accent_split_folder, LANG, "speakers.pth"))
    for wav_path, name_emb in temp.items():
        spk = name_emb["name"]
        emb = name_emb["embedding"]
        if spk not in SPK2WAV:
            SPK2WAV[spk] = {}
        SPK2WAV[spk][wav_path] = emb
        embs.append(emb)
        embs_info.append((LANG, spk, wav_path))
    SPK_EMBs[LANG] = SPK2WAV

downdimensioner = TSNE(n_components=2)
downdimensioned_embs = downdimensioner.fit_transform(np.array(embs))

downdimensioned_SPK_EMBs = {}
for downdimensioned_emb, (LANG, spk, wav_path) in zip(downdimensioned_embs, embs_info):
    if LANG not in downdimensioned_SPK_EMBs:
        downdimensioned_SPK_EMBs[LANG] = {}
    if spk not in downdimensioned_SPK_EMBs[LANG]:
        downdimensioned_SPK_EMBs[LANG][spk] = {}
    downdimensioned_SPK_EMBs[LANG][spk][wav_path] = downdimensioned_emb

plt.figure(figsize=(6,5))
plt.clf()

colormap_name = 'tab20'
colormap = plt.get_cmap(colormap_name)

colors = colormap(np.linspace(0, 1, len(SPK_EMBs)))

for i, (lang, spk2wav) in enumerate(downdimensioned_SPK_EMBs.items()):
    lang_spks_emb = []
    for spk, path_emb in tqdm(spk2wav.items()):

        # take the mean of all wavs by one speaker
        spk_wavs_emb = np.array(list(path_emb.values()))
        spk_emb_mean = np.mean(spk_wavs_emb, axis=0)
        lang_spks_emb.append(spk_emb_mean)

        # take all wavs by one speaker
        # lang_spks_emb += list(path_emb.values())

    lang_spks_emb = np.array(lang_spks_emb)
    plt.scatter(lang_spks_emb[:, 0], lang_spks_emb[:, 1], color=colors[i], alpha=1, linewidths=1, s=10)
    lang_spk_emb_mean = np.mean(lang_spks_emb, axis=0)
    plt.plot(lang_spk_emb_mean[0], lang_spk_emb_mean[1], marker="o", markerfacecolor=colors[i], markeredgecolor="k", markersize=8)
    plt.text(lang_spk_emb_mean[0], lang_spk_emb_mean[1], lang, multialignment="right", size="10", color="black")

plt.xlabel('t-SNE-1',fontsize=14)
plt.ylabel('t-SNE-2',fontsize=14)

plt.savefig("./log/tsne_speaker_embeddings.png")
# plt.savefig("./log/tsne_speaker_embeddings_allwavs.png")
plt.show()
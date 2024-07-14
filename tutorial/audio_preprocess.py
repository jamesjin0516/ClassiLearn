import librosa
import matplotlib.pyplot as plt
import numpy as np

sig1, rate1 = librosa.load("audio.wav", mono=True)
sig2, rate2 = librosa.load("audio_estereo.wav")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].plot(np.arange(sig1.shape[0]) / rate1, sig1)
axes[0].set_title("audio mono", fontsize=20)
axes[1].plot(np.arange(sig2.shape[0]) / rate2, sig2)
axes[1].set_title("audio estereo", fontsize=20)
axes[0].tick_params(which="both", labelsize=20)
axes[1].tick_params(which="both", labelsize=20)
fig.savefig("audio_raw.png")

sig1_norm = (sig1 - np.mean(sig1)) / np.max(np.absolute(sig1))
sig2_norm = (sig2 - np.mean(sig2)) / np.max(np.absolute(sig2))
fig_norm, axes_norm = plt.subplots(1, 2, figsize=(20, 10))
axes_norm[0].plot(np.arange(sig1_norm.shape[0]) / rate1, sig1_norm)
axes_norm[0].set_title("audio mono", fontsize=20)
axes_norm[1].plot(np.arange(sig2_norm.shape[0]) / rate2, sig2_norm)
axes_norm[1].set_title("audio estereo", fontsize=20)
axes_norm[0].tick_params(which="both", labelsize=20)
axes_norm[1].tick_params(which="both", labelsize=20)
fig_norm.savefig("audio_preprocessed.png")

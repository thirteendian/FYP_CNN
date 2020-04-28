import matplotlib.pyplot as plt
from pickle import load, dump
import address_config
import numpy as np
from os.path import join
from madmom.audio.filters import MelFilterbank
from madmom.audio.signal import FramedSignal, Signal
from madmom.audio.spectrogram import (FilteredSpectrogram,LogarithmicSpectrogram)
from madmom.audio.stft import ShortTimeFourierTransform

data=load(open(r'D:\projectRESEARCH\CNN\softonsetdetection\cache\combined_pure_violin\cache.pkl','rb'))
onsets=load(open('D:\projectRESEARCH\CNN\softonsetdetection\models\combined_pure_violineval_cache.pkl','rb'))
def process_signal(signal,framesize):
    """
    Using madmom package to process file
    :return: processed files, normalized logFBE
    """
    frames = FramedSignal(signal,frame_size=framesize,fps=100)
    stft = ShortTimeFourierTransform(frames)
    filter_bank_energy = FilteredSpectrogram(stft,
                                             filterbank = MelFilterbank,
                                             num_bands = 80,
                                             fmin=27.5,
                                             fmax=16000,
                                             norm_filters=True,
                                             unique_filters=False)
    log_filter_bank_energy = LogarithmicSpectrogram(filter_bank_energy,
                                                    log=np.log,
                                                    add=np.spacing(1))
    return log_filter_bank_energy

def generator(i):
    signal = Signal(
        join(r'D:\projectRESEARCH\CNN\softonsetdetection\data\combined_pure_violin\audio', data[i+6].an + '.flac'),
        sample_rate=44100, num_channels=1)
    signal_a = process_signal(signal, 1024)
    onset_tp = 100 * onsets[i].tp
    onset_fp = 100 * onsets[i].fp
    onset_fn = 100 * onsets[i].fn
    plt.figure(figsize=(20, 5))
    plt.imshow(signal_a.T, origin='lower', aspect='auto')
    for xt in onset_tp:
        plt.axvline(x=xt, color='r', lw=2.5)
    for xf in onset_fp:
        plt.axvline(x=xf, color='m', lw=2.5)
    for xn in onset_fn:
        plt.axvline(x=xn, color='r', lw=2.5, ls='--')
    plt.savefig(join(r'D:\projectRESEARCH\CNN\softonsetdetection\figure\audio_map',data[i].an+'.png'))
    plt.show()
#
# for i in range(len(data)):
generator(2)


# plt.figure()
#     img_1 = plt.imshow(signal.T[:400], origin='lower', aspect='auto')
#     plt.xlabel('frame')
#     plt.ylabel('log_mel freq')
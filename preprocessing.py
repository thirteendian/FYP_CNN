########################################################################################################################
# Preprocessing
# ----------------------------------------------------------------------------------------------------------------------
# This file is the preprocessing of data
# First check if there are processed data cache
# Second if not processed before then process, if yes then import cache
#
########################################################################################################################



import address_config
from collections import namedtuple

from os import listdir
from os.path import join,basename,splitext,exists
from pickle import load,dump
import random
import numpy as np

from madmom.audio.filters import MelFilterbank
from madmom.audio.signal import FramedSignal, Signal
from madmom.audio.spectrogram import (FilteredSpectrogram,LogarithmicSpectrogram)
from madmom.audio.stft import ShortTimeFourierTransform





def list_of_audio(data_dir):
    audio_dir = join(data_dir,'audio')
    # return name format of "../audio/files"
    return [join(audio_dir,f) for f in sorted(listdir(audio_dir))]

def list_of_anno(data_dir):
    ann_dir = join(data_dir,'annotations','onsets')
    # return name format of "../annotations/onsets/onsetfiles"
    return [join(ann_dir,f) for f in sorted(listdir(ann_dir))]



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

def process_padding(filename):
    """
    read signal and process signal for SINGLE file
    change format to (n,80,3)
    add padding of extra 14
    :return: processed file with 14 padding ready for slice single
    D[80,3]
    """
    # read the data
    signal = Signal(filename,sample_rate=44100,num_channels=1)
    D=[process_signal(signal,f) for f in [2048,1024,4096]]

    # change (3, 80) into CNN format (80, 3)
    D=np.dstack(D)

    #Add padding by repeating the same frames, because one slice is 15, surely need extra 14 on each side
    leftpadding = np.repeat(D[:1],7,axis=0)
    rightpadding= np.repeat(D[-1:],7,axis=0)
    D=np.concatenate((leftpadding,D,rightpadding))
    return D

def process_all_files(audiofiles):
    """
    Process all files
    :return: all of the processed files
    """
    n=len(audiofiles)
    for i,file_address in enumerate(audiofiles):
        args=(i+1,n,basename(file_address))
        print('[%3d/%3d]%-90s'%args,end='',flush=True)
        yield process_padding(file_address)
        print('AUDIO FILE PROCESS DONE')

def process_annotation(anns,frame_number):
    """
    change annotation(second) to binary annotation denotes the position
    :return: [0....1....0...1...], 1 denotes onset, 0 denotes nononset
    """
    frame_number-=14

    #because fps=100, the second number * 100 is the onset position in the series
    position=(anns * 100).astype(int)

    # each onset position give 1, nononset give 0
    binary_annotations = np.zeros(frame_number)
    binary_annotations[np.unique(position)] = 1

    #This two line add additional for soft onset
    binary_annotations[np.unique(position+1)] = 1
    binary_annotations[np.unique(position-1)] = 1
    return binary_annotations

AudioSample = namedtuple('AudioSample',['pd','ba','ra','an'])
"""
corresponding to 

pd:Processed_data
ba:binary_annotation
ra:raw_annotation
an:audio_name

"""
def preprocess_data(seed, data_address):
    """
    seperate the

    :param seed:
    :param data_address:
    :return:
    """
    audiofiles = list_of_audio(data_address)
    annofiles = list_of_anno(data_address)
    #Processed files

    # PD=[[n,80,3],[n,80,3],....,[n,80,3]]
    PD = list(process_all_files(audiofiles))

    #normalized

    # [[anno1],[anno2],[anno3]]
    RA = [np.loadtxt(f) for f in annofiles]

    # [[0...1...0],[1...0...1],[0...1...1]....]
    BA = [process_annotation(ra,len(pd)) for (ra,pd) in zip(RA,PD)]

    # [[filenames],[filenames],...,[filenames]]
    AN =[splitext(basename(n))[0] for n in audiofiles]

    D=[AudioSample(pd,ba,ra,an) for pd,ba,ra,an in zip(PD,BA,RA,AN)]

    # shuffle the order to make higher robust
    random.seed(seed)
    random.shuffle(D)

    # Normalized the data in z-score(Standard Score), z=(x-mu)/sigma, mu is mean, sigma is standard deviation
    # normally the value from +-5~6
    all_data = np.concatenate([d.pd for d in D])
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    D = [AudioSample((d.pd - mean) / std, d.ba, d.ra, d.an) for d in D]
    return D

def cache_access(cache_address, function_preprocess_data):
    """
    Using pickle to laod or generate python object of processed data
    If object exist, loading. If not, generating
    :return: processed data(cache)
    """
    print('TRYING TO LOAD CACHE....')
    if exists(cache_address):
        print('CACHE FOUND. LOADING CACHE FROM %s'%cache_address)
        return load(open(cache_address,'rb'))
    else:
        print('CACHE NOT FOUND. GENERATING CACHE TO %s'%cache_address)
        data=function_preprocess_data()
        dump(data,open(cache_address,'wb'),protocol=2)
        return data
def data_access():
    data_address=address_config.data_dir
    cache_address=address_config.cache_dir
    seed = address_config.seed
    saved_cache=join(cache_address,'cache.pkl')

    function_preprocess_data= lambda:preprocess_data(seed,data_address)
    D = cache_access(saved_cache,function_preprocess_data)

    return [D[i::8] for i in range(8)] # divide the audio files to 8 parts

#D=data_access()

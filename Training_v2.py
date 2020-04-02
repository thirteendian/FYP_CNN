"""
========================================================================================================================
YUXUAN YANG 1976844 University of Birmingham Final Year Project
March 2020

This is the modified version of Training and predicting
"""

import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
from scipy.fftpack import fft
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from sklearn.preprocessing import normalize
import sklearn
import sklearn.metrics

import keras.backend as K
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras

"""
=======================================================================================================================
"""
FRAME_LENGTH = 1024  # Each Frame contains 1024 samples
FRAME_STEP = 512  # Each Frame overlap 512 samples
SLICE_LENGTH = 15  # Each Slice contains 15 frames as img_col
CHANNEL = 1
ONSET_CLASSES = 2
# PITCH_CLASSES =
AUDIO_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week8\\onsets_ISMIR_2012\\audio'
ANNOTATION_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week8\\onsets_ISMIR_2012\\annotations\\onsets'
DATA_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week8\\onsets_ISMIR_2012\\splits'


def read_le_metadata(data_address, filename):
    df = pd.read_csv(data_address + '\\' + filename, header=None, names=['f'])
    return df


def read_le_file(audio_address, annotation_address):
    """

    Read audio files .wav and annotation .csv
    :param audio_address: audio file address
    :param annotation_address: annotation file address
    :return:
        sample_rate:
        data:
        signal_length:
        number_of_frame:
        df: Panda data frame

    """
    sample_rate, data = wavfile.read(audio_address)
    #data, sample_rate = librosa.load(audio_address, sr=44100)  # make sure that sample rate are all same
    df = pd.read_csv(annotation_address, header=None, names=['onset'])
    signal_length = len(data)
    number_of_frame = int(np.ceil((signal_length - FRAME_LENGTH) / FRAME_STEP))  # np.ceil down to integer
    return sample_rate, data, signal_length, number_of_frame, df


def divide_le_frame(data, frame_step, frame_length, signal_length, number_of_slice):
    """
    Transfer the audio data to frame form, each length 1024 samples with 512 overlap
    :param number_of_slice:
    :param data: Audio data
    :param frame_step: Constant
    :param frame_length: Constant
    :param signal_length: Dimension of frame matrix
    :param number_of_slice: Number of
    :return: ndarray (number_of_slice*1024)
    """
    extended_signal_length = number_of_slice * frame_step + frame_length

    extended_zeros = np.zeros(extended_signal_length - signal_length)  # number of zeros at the end to complete matrix
    extended_data = np.append(data, extended_zeros)

    index_of_frames = np.tile(np.arange(0, frame_length), (number_of_slice, 1)) \
                      + np.tile(np.arange(0, number_of_slice * frame_step, frame_step), (frame_length, 1)).T
    framedata = extended_data[index_of_frames.astype(np.int32, copy=False)]
    return framedata


def calculate_le_mel_spec(frame_length, sample_rate, framedata):

    framedata_windowed = np.hamming(frame_length) * framedata

    NFFT = (frame_length / sample_rate) * sample_rate
    frame_magnitude = np.absolute(np.fft.rfft(framedata_windowed, NFFT))
    framedata_power_spectrum = (1 / NFFT) * (frame_magnitude ** 2)

    # high_freq_mel = 2595 * (np.log10(1 + (sample_rate / 2) / 700))
    melfilter = mel(sample_rate, NFFT, 80, 27.5, 16000)
    framedata_mel_filtered = np.dot(np.abs(framedata_power_spectrum), melfilter.T)

    framedata_mel_log = 20 * np.log10(framedata_mel_filtered + 0.1)

    framedata_normalized = librosa.util.normalize(framedata_mel_log)
    return framedata_normalized


def create_les_labels_et_onset(number_of_frame, df, framedata_normalized):

    frametime = np.arange(number_of_frame) * (1024.0 / 44100.0)  # a scale vector of each frame in time domain
    # [1*1024/44100, 2*1024/44100,..., number_of_frame*1024/44100]

    saved_column = df.onset.to_numpy()  # Onset column to numpy array
    onset_index = []  # where onset happened
    for i in range(len(saved_column)):
        onset_index.append(np.argmin(np.abs(frametime - saved_column[i])))  # find nearest onset frame
        # np.argmin return the minimum value
        # onset_index.append(np.argmin(np.abs(frametime + 0.025 - saved_column[i])))  # tolerance +25ms
        # onset_index.append(np.argmin(np.abs(frametime - 0.025 - saved_column[i])))  # tolerance -25ms

    label = [0] * number_of_frame
    for i in onset_index:
        label[i] = 1

    # Cut the useless end bunch of 0
    label = label[:onset_index[-1] + 1]
    framedata_normalized = framedata_normalized[:onset_index[-1] + 1]

    return framedata_normalized, label, onset_index

"""
def generate_les_pictures(framedata_normalized, label):
    img_data = []
    label_data = []
    for i in range(8, len(framedata_normalized) - 8, 1):
        img_data.append(framedata_normalized[i - 8:i + 7].T)
        label_data.append(label[i])  # only consider the middle frame as label of this picture

    img_data = np.asarray(img_data)
    label_data = np.asarray(label_data)
    img_data = img_data.astype('float32')
    label_data = label_data.astype('int64')

    if CHANNEL == 1:
        if K.image_data_format() == 'channels_last':
            img_data = np.expand_dims(img_data, axis=3)  # Tensorflow form
        else:
            img_data = np.expand_dims(img_data, axis=1)  # Theano form(img,)
    return img_data, label_data
"""

def generate_les_pictures(framedata_normalized, label):
    img_data = []
    label_data = []
    for i in range(8, len(framedata_normalized) - 8, 1):
        img_data.append(framedata_normalized[i - 8:i + 7].T)
        label_data.append(label[i])  # only consider the middle frame as label of this picture
    # find onset position in slice
    index_onset = [i for i in range(len(label_data)) if label_data[i] == 1]  # all onset image
    index_nononset = [i for i in range(len(label_data)) if label_data[i] != 1]  # other

    # classify onset pair and nononset pair
    img_onset = [img_data[i] for i in index_onset]
    img_nononset = [img_data[i] for i in index_nononset]
    label_onset = [label_data[i] for i in index_onset]
    label_nononset = [label_data[i] for i in index_nononset]

    img_onset = np.asarray(img_onset)
    img_nononset = np.asarray(img_nononset)
    label_onset = np.asarray(label_onset)
    label_nononset = np.asarray(label_nononset)

    img_onset = img_onset.astype('float32')
    img_nononset = img_nononset.astype('float32')
    label_onset = label_onset.astype('int64')
    label_nononset = label_nononset.astype('int64')

    # img_data.shape >>>(number_of_slice, 80, 15)   -------3D array

    # check keras background form

    if CHANNEL == 1:
        if K.image_data_format() == 'channels_last':
            img_onset = np.expand_dims(img_onset, axis=3)  # Tensorflow form
            img_nononset = np.expand_dims(img_nononset, axis=3)
        else:
            img_onset = np.expand_dims(img_onset, axis=1)  # Theano form(img,)
            img_nononset = np.expand_dims(img_nononset, axis=1)
    # img_data.shape >>>(number_of_slice,80,15,1)   -------4D array
    return img_onset, label_onset, img_nononset, label_nononset

def balance_les_data(img_onset, label_onset, img_nononset, label_nononset):
    img_nononset, label_nononset = sklearn.utils.shuffle(img_nononset, label_nononset, random_state=2)
    img_nononset = img_nononset[:len(img_onset)+50]
    label_nononset = label_nononset[:len(label_onset)+50]
    img_balanced = np.concatenate((img_onset, img_nononset), axis=0)
    label_balanced = np.concatenate((label_onset, label_nononset), axis=0)
    return img_balanced, label_balanced

def train_data_processing(wavename, annotationname):
    '''
    The meta data should have the form of
    wavefile   annotation
    wave1.wav  xxx.csv
    wave2.wav  xxx.csv
    ...

    '''
    sample_rate, data, signal_length, number_of_frame, df = \
        read_le_file(AUDIO_ADDRESS + '\\' + wavename, ANNOTATION_ADDRESS + '\\' + annotationname)
    framedata = divide_le_frame(data, FRAME_STEP, FRAME_LENGTH, signal_length, number_of_frame)
    framedata_normalized = calculate_le_mel_spec(FRAME_LENGTH, sample_rate, framedata)
    framedata_normalized_labelled, label, onset_index = create_les_labels_et_onset(number_of_frame, df,
                                                                                   framedata_normalized)
    """
    plt.figure()
    img_1 = plt.imshow(framedata_normalized_labelled.T[:400], origin='lower', aspect='auto')
    plt.xlabel('frame')
    plt.ylabel('log_mel freq')
    """
    img_onset, label_onset, img_nononset, label_nononset = generate_les_pictures(framedata_normalized_labelled, label)
    img_train, label_train = balance_les_data(img_onset, label_onset, img_nononset, label_nononset)
    return img_train, label_train


"""
=======================================================================================================================
"""
file_name = os.listdir(DATA_ADDRESS)
img_train = np.zeros(shape=(0, 80, 15, 1))  # training img
label_train = np.zeros(shape=(0))  # training lable
for f in file_name[0:1]:
    dfmeta = read_le_metadata(DATA_ADDRESS, f)
    for i in range(len(dfmeta.f)):
        img_buffer, label_buffer = train_data_processing(dfmeta.f[i] + '.wav', dfmeta.f[i] + '.csv')
        img_train = np.concatenate((img_train, img_buffer))
        label_train = np.concatenate((label_train, label_buffer))
img_train = img_train.astype('float32')
label_train = label_train.astype('int64')


# -----------------------------------------------------------------------------------------------------------------------

def split_les_data_et_labels(img_train, label_train):
    img_data, label_data = sklearn.utils.shuffle(img_train, label_train, random_state=2)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(img_data, label_data, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = split_les_data_et_labels(img_train, label_train)
bench, height, width, channels = X_train.shape

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=10, kernel_size=(3, 7), strides=(1, 1), padding="VALID", activation="tanh",
                        input_shape=[height, width, channels]),
    keras.layers.MaxPooling2D(pool_size=(3, 1), strides=None, padding="VALID"),
    keras.layers.Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), padding="VALID", activation="tanh"),
    keras.layers.MaxPooling2D(pool_size=(3, 1), strides=None, padding="VALID"),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="mean_squared_error",
              metrics=['accuracy'],
              optimizer= keras.optimizers.SGD(learning_rate=0.01,momentum=0.9))  # 0.05 0.45

history = model.fit(X_train, Y_train,
                    batch_size=256, epochs=100, verbose=1, validation_data=(X_test, Y_test))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

# predictions = model.predict(X_test)

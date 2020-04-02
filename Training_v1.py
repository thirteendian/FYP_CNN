"""
=======================================================================================================================
YUXUAN YANG 1976844 University of Birmingham Final Year Project
March 2020
This is the original version of the data
-----------------------------------------------------------------------------------------------------------------------
There are several things for this project:

Comparing with finding items in picture:
https://www.lunaverus.com/cnn
1.a fundamental frequency is composed of harmonics at multiples of that fundamental frequency, which tend to
decrease in "Amplitude" as you go up.
2.Nearby harmonics can result in amplitude "beating", algorithm need to take this into consideration.
To reduce interference effects, increasing the Q factor
3.It's impossible to process whole spectrum at once so we need "Firstly Detect the Onset position", A seperate CNN
need to be trained to do onset detection. (Lunaverus.com said it's ignored the offsets, assume that every note ends
when the next note begins)
4. For creating spectrogram, some use "constant Q transform"
a constant frequency to bandwidth ratio with 4 frequency bins per note. Work well for CNN as same distance between
1st 2nd 3rd ... harmonics, no need for fully connected layers.
=======================================================================================================================
"""
import librosa
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
AUDIO_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week2\\codex\\data\\wavefiles\\'
ANNOTATION_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week2\\codex\\data\\annotations\\'
DATA_ADDRESS = 'D:\\projectRESEARCH\\CNN\\Aut_Week2\\codex\\data\\'


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
    # data, sample_rate = librosa.load(audio_address, sr=44100)  # make sure that sample rate are all same
    df = pd.read_csv(annotation_address)
    signal_length = len(data)
    number_of_frame = int(np.ceil((signal_length - FRAME_LENGTH) / FRAME_STEP))  # np.ceil down to integer
    return sample_rate, data, signal_length, number_of_frame, df


# -----------------------------------------------------------------------------------------------------


"""
=======================================================================================================================
"""


def divide_le_frame(data, frame_step, frame_length, signal_length, number_of_frame):
    """
    Transfer the audio data to frame form, each length 1024 samples with 512 overlap
    :param number_of_frame:
    :param data: Audio data
    :param frame_step: Constant
    :param frame_length: Constant
    :param signal_length: Dimension of frame matrix
    :param number_of_slice: Number of
    :return: ndarray (number_of_slice*1024)
    """
    extended_signal_length = number_of_frame * frame_step + frame_length

    extended_zeros = np.zeros(extended_signal_length - signal_length)  # number of zeros at the end to complete matrix
    extended_data = np.append(data, extended_zeros)

    index_of_frames = np.tile(np.arange(0, frame_length), (number_of_frame, 1)) \
                      + np.tile(np.arange(0, number_of_frame * frame_step, frame_step), (frame_length, 1)).T
    """
    np.tile(a, (b,c)) repeat a vector c times in one row, and repeat this row b times as a matrix.
    [0,1,2,3,...,1024]              [0,0,0,0,...,0]
    [0,1,2,3,...,1024]              [1*256,1*256,1*256,1*256,...,1*256]
    [0,1,2,3,...,1024]      +       [2*256,2*256,2*256,2*256,...,2*256]
    ...                             ...
    [0,1,2,3,...,1024]              [number_of_slice*256,number_of_slice*256,number_of_slice*256,...]
    (number_of_slice*1024)          (number_of_slice*1024)
    The above line create the divided index for data
     
    """
    framedata = extended_data[index_of_frames.astype(np.int32, copy=False)]
    return framedata


# ----------------------------------------------------------------------------------------------------------------------


"""
=======================================================================================================================
Window, NFFT, FFT, power spectrum
Mel filter
    80-band Mel filter from 27.5Hz to 16kHz
    normalized
=======================================================================================================================
"""


def calculate_le_mel_spec(frame_length, sample_rate, framedata):
    """
    Process: NFFT->Power Spectrum->Mel Filter->Logarithm->Normalization
    :param frame_length: constant
    :param sample_rate: constant
    :param framedata: input data (with each framed)
    :return:
    """
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


"""
=======================================================================================================================
Onset Annotation Translation
For 1024/44100 (second) as one frame
The position of 1 in label_Y = The round_down([onset]*44100/1024)
for round_down function using int()
with tolerance of around +-25ms, the neighbour of label also be 1
=======================================================================================================================
"""


def create_les_labels_et_onset(number_of_frame, df, framedata_normalized):
    """
    Read the data from csv file to create labels for training
    basically translate sec unit to index unit in vector
    :param number_of_frame: whole interval length
    :param df: csv portal
    :param framedata_normalized: data input
    :return: label
    """
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
------------------------------------------------------------------------------------------------------------------------
"""

# Plot the dataset


"""

=======================================================================================================================
This will transfer audio data to image data with picture form
-------------
Channel           -- According to paper 2014_icassp, using three different widows as same as RGB
Width (img_rows)  -- Spectrum diagram's width exactly as Picture's width
Length(img_cols)  -- Spectrum diagram's Length exactly as Picture's Length
-------------
img_cols = 80
img-rows = 15
Channel  = 1
overlap step is 1
|
|
80
|
|___15___

The label corresponding to the picture are the label of middle frame: frame 8
there will be around 50ms tolerance, which consider the neighbor frame of onset are also onset picture
[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
[     i-7    ] i [       i+7        ]

The format is "channels_last" format in keras.backend

So finally the dataset will be a 4D array
(number_of_slices, number_of_rows, number_of_columns, channel)
corresponding to 
(batch_size, height, width, depth)

remark: the output of CNN will also be a 4D array
=======================================================================================================================
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



def generate_les_test_pictures(framedata_normalized, label):
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
            img_data = np.expand_dims(img_data, axis=1)  # Theano form
    return img_data, label_data


"""
=======================================================================================================================


"""


def balance_les_data(img_onset, label_onset, img_nononset, label_nononset):
    img_nononset, label_nononset = sklearn.utils.shuffle(img_nononset, label_nononset, random_state=2)
    img_nononset = img_nononset[:len(img_onset)+50]
    label_nononset = label_nononset[:len(label_onset)+50]
    img_balanced = np.concatenate((img_onset, img_nononset), axis=0)
    label_balanced = np.concatenate((label_onset, label_nononset), axis=0)
    return img_balanced, label_balanced


"""
-----------------------------------------------------------------------------------------------------------------------
"""


def train_data_processing(wavename, annotationname):
    '''
    The meta data should have the form of
    wavefile   annotation
    wave1.wav  xxx.csv
    wave2.wav  xxx.csv
    ...

    '''
    sample_rate, data, signal_length, number_of_frame, df = \
        read_le_file(AUDIO_ADDRESS + wavename, ANNOTATION_ADDRESS + annotationname)
    framedata = divide_le_frame(data, FRAME_STEP, FRAME_LENGTH, signal_length, number_of_frame)
    framedata_normalized = calculate_le_mel_spec(FRAME_LENGTH, sample_rate, framedata)
    framedata_normalized_labelled, label, onset_index = create_les_labels_et_onset(number_of_frame, df,
                                                                                   framedata_normalized)
    plt.figure()
    img_1 = plt.imshow(framedata_normalized_labelled.T[:400], origin='lower', aspect='auto')
    plt.xlabel('frame')
    plt.ylabel('log_mel freq')

    img_onset, label_onset, img_nononset, label_nononset = generate_les_pictures(framedata_normalized_labelled, label)
    img_train, label_train = balance_les_data(img_onset, label_onset, img_nononset, label_nononset)
    return img_train, label_train, img_onset,label_onset,data


def test_data_processing(wavename, annotationname):
    sample_rate, data, signal_length, number_of_frame, df = \
        read_le_file(AUDIO_ADDRESS + wavename, ANNOTATION_ADDRESS + annotationname)
    framedata = divide_le_frame(data, FRAME_STEP, FRAME_LENGTH, signal_length, number_of_frame)
    framedata_normalized = calculate_le_mel_spec(FRAME_LENGTH, sample_rate, framedata)
    framedata_normalized_labelled, label, onset_index = create_les_labels_et_onset(number_of_frame, df,
                                                                                   framedata_normalized)
    img_data, label_data = generate_les_test_pictures(framedata_normalized_labelled, label)
    return img_data, label_data, framedata_normalized_labelled, onset_index


df = pd.read_csv(DATA_ADDRESS + 'metadata.csv')
img_train = np.zeros(shape=(0, 80, 15, 1))  # training img
label_train = np.zeros(shape=(0))  # training lable
img_for_pred = np.zeros(shape=(0, 80, 15, 1))  # img for prediction
label_for_exam = np.zeros(shape=(0))  # img for training
for i in range(len(df.wavefile)):
    img_buffer, label_buffer, img_onset,label_onset,data= train_data_processing(df.wavefile[i], df.annotation[i])
    img_train = np.concatenate((img_train, img_buffer))
    label_train = np.concatenate((label_train, label_buffer))
# __________________________________________________________________________-
img_for_pred, label_for_exam, audio_map, onset_index_for_exam = \
    test_data_processing(df.test_wavefile[0], df.test_annotation[0])

img_train = img_train.astype('float32')
label_train = label_train.astype('int64')

img_for_pred = img_for_pred.astype('float32')
label_for_exam = label_for_exam.astype('int64')
"""
split the data to validation and training
test_size give to 0.2
Here we set random_state a constant that the result can repeat the same result
"""


def split_les_data_et_labels(img_train, label_train):
    img_data, label_data = sklearn.utils.shuffle(img_train, label_train, random_state=2)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(img_data, label_data, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = split_les_data_et_labels(img_train, label_train)

"""
=======================================================================================================================
CNN Model

-------------
Filters-- the number of input Channels of next layer
Kernel_size-- window size
Strides-- window jumping step
Padding-- adding Zeros around the input, "valid" means no padding, "same" means add same length padding
-------------
ATTENTIOM: if its a binary classification, using "sigmoid", If it's multi classification, using "softmax"
The manual layer joint in keras layer using Lambda for example:
keras.layers.Lambda(
    lambda X: tf.nn.max_pool(X, ksize=(batch,height,width,depth),strides=(batch,height,width,depth),padding="valid"))

Structure

1-channel input[15x80]
||
========================con[7*3][tanh](overlap,strides=(1,1)) (height,width)
||
10-feature map[9x78]
||
========================maxpool[1x3](no overlap,strides = (1,3))
||
10-feature map[9x26]
||
========================con[3*3][tanh](overlap,strides=(1,1))  (height,width)
||
20-feature map[7x24]
||
=========================maxpool[1x3](no overlap,strides = (1,3))
||
20-feature map[7x8]
||
=========================Dense[256]
||
=========================Dense[1] "sigmoid"

SGD: learning rate = 0.05, momentum = 0.45

=======================================================================================================================
"""
"""
F-1 scores

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    For f1 as f1_score definition, for f1_loss as the diffrentialable loss function according to f1
    For a 50ms correction tolerance, redefine the y_true, as around 1024/44100 a frame second, then 0.05/(1024/44100)
    with around 2.15 frame a duration
"""


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  # number of corrected prediction
    # NOt presice because number of corrected may wrongly calculated multiple times,
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  # number of inserted onsets
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  # number of deleted onsets

    p = tp / (tp + fp + K.epsilon())  # Precision
    r = tp / (tp + fn + K.epsilon())  # Recall

    f1 = 2 * p * r / (p + r + K.epsilon())  # F1 score
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):  # minimizing 1-f is equial to maximizing f, we want the maximum f in loop#
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


"""##############################################################"""

bench, height, width, channels = X_train.shape

# initialize the model
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=10, kernel_size=(3, 7),
                        strides=(1, 1), padding="VALID",
                        activation="tanh",
                        input_shape=[height, width, channels]),
    keras.layers.MaxPooling2D(pool_size=(3, 1),
                              strides=None, padding="VALID"),
    keras.layers.Conv2D(filters=20, kernel_size=(3, 3),
                        strides=(1, 1), padding="VALID",
                        activation="tanh"),
    keras.layers.MaxPooling2D(pool_size=(3, 1), strides=None,
                              padding="VALID"),
    keras.layers.Flatten(),

    keras.layers.Dense(256, activation="relu"),

    keras.layers.Dense(1, activation="sigmoid")
])
"""
Initialize optimizer
"""
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.02)
"""
Initialize Loss Function
The Binarycrossentropyloss(for binary classification) or Crosscategorical entropy(Multi-classification) is not recommended
 https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
The best Loss function is the metric itself. Macro F1-score. To make it differentiable(that be able to be a loss function)

As previous defined that f1_loss is the loss function
"""
loss_fn = f1_loss
#es = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=70,verbose=0, mode='auto') callbacks=[es]

model.compile(loss="mean_squared_error",
              metrics=['accuracy'],
              optimizer=keras.optimizers.SGD(learning_rate=0.02, momentum=0.9))  # 0.05 0.45

history = model.fit(X_train, Y_train,
                    batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test))
"""
batch_size = 10
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
epochs = 3
for epoch in range(epochs):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:  # operation flow
            # Run the forward pass of the layer. The operation are going to be recorded
            logits = model(x_batch_train, training=True)
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)
            # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to 
            the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # Log every 200 batches.
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

"""

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

"""
predictions = model.predict(mini_bench2)

"""
predictions = model.predict(img_for_pred)
"""

onset_position = []
for i in range(len(predictions)):
    onset_flag=0
    if predictions[i]>0.9:
        onset_position.append(i)
onset_position= onset_position*(1024/44100)
from sklearn.metrics import precision_score, recall_score
precision=precision_score(df2.onset, onset_position)
recall=recall_score(df2.onset, onset_position)
F_score= (2*precision*recall)/(precision+recall).;

"""

# predictions = model.predict(img_data2)
predictions = model.predict(X_test)
for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i]=1
    else:
        predictions[i]=0
x=[i for i in range(len(Y_test)) if Y_test[i]==1]
y=[i for i in range(len(predictions)) if predictions[i]==1]
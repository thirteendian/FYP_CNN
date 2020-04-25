#####################################################################
# Cnn
# -------------------------------------------------------------------
# This file is the Cnn
# Define train() function
# Create CNN model
# Using model.fit_generator()
# Data generator "ArchSequence(Keras.utils.Sequence)"
# heritage from Keras.utils.Sequence
# created by Yuxuan Yang 23 April 2020
#####################################################################
import address_config
from os.path import join
import numpy as np
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt

from pickle import dump

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

import multi_model
# Tensorflow 2.1
# CUDA Toolkit 10.1 update2[Aug 2019]
# cudnn-10.1


def model():
    m = tf.keras.Sequential()
    m.add(Conv2D(10, (7, 3), input_shape=(15, 80, 3),
                 padding='valid', activation='relu'))
    m.add(MaxPooling2D(pool_size=(1, 3)))
    m.add(Conv2D(20, (3, 3), input_shape=(9, 26, 10),
                 padding='valid', activation='relu'))
    m.add(MaxPooling2D(pool_size=(1, 3)))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(256, activation='sigmoid'))
    m.add(Dense(1, activation='sigmoid'))

    optimizer = SGD(lr=0.05, momentum=0.8, clipvalue=5)
    m.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['binary_accuracy'])
    return m


# define generator heritage from keras.utils.Sequence
class ArchSequence(tf.keras.utils.Sequence):
    """
    This class is heritage from Sequence data generator
    """

    def __init__(self, D, batch_size=256, shuffle=True):
        """
        D:          List of AudioSequence type
        batch_size: as defined
        n_samples:  total number of onsets samples in all slice
        indices:    index of total onset samples
        """
        self.D = D
        self.batch_size = batch_size
        self.n_samples = sum(len(d.ba) for d in self.D)
        self.indices = np.arange(self.n_samples)
        self.shuffle = shuffle

    def __len__(self):
        """
        number of batches per epoch
        __len__:        return number of batches per epoch,
                        [samples/batch_size]
        __getitem__:    generator executes
        """
        return int(np.ceil(self.n_samples / self.batch_size))

    def get_sample(self, index_total):
        """
        n_frames:   total frame number, pd in AudioSample is 14 extended
        ----------------------------------------------------------------
        for the index appear in this function:
        index_total: total index of all onset samples
        index_start: start index flag among index_total
        index_total-index_start: the start index of each slice
        index_start+=n_frames: update start index
        """
        index_start = 0
        for d in self.D:
            n_frames = len(d.ba)

            # make sure audio data is padding added
            assert len(d.pd) == n_frames + 14
            if index_start + n_frames > index_total:
                # For [d1:dn] each slice start back from beginning
                ofs = index_total - index_start
                # map of 15 frames
                return d.pd[ofs:ofs + 15], d.ba[ofs]
            index_start += n_frames
        raise IndexError('INDEX `%d` WRONG...' % index_total)

    def __getitem__(self, index):
        """
        indexes:        [n:n*batch_size], final loop use min() to stop exceed

        """
        indexes = self.indices[index * self.batch_size:min((index + 1) * self.batch_size, self.n_samples)]
        samples = [self.get_sample(i) for i in indexes]  # [([15],[0/1]),...]
        # intimate of CNN on picture
        pictures = np.array([s[0] for s in samples])
        labels = np.array([s[1] for s in samples])
        return pictures, labels

    def on_epoch_end(self):
        """
        How to shuffle after each epoch for data generator
        https://github.com/keras-team/keras/issues/9707
        """
        if self.shuffle == True:
            np.random.shuffle(self.indices)


def preparefortrain(data, test_idx, epochs):
    """
    validation set is next to current slices
    training set is all left slices
    so no current slice is used for training/validation
    it's left for test and evaluation
    """
    # define model saved path of each slice
    model_dir = address_config.model_dir
    slice_dir = join(model_dir, '%02d' % test_idx)

    n = len(data)
    # validation sets always be the next slice
    # when test is the last, val change back to first
    # like the Ring Buffer
    val_idx = (test_idx + 1) % n

    test = ArchSequence(data[test_idx])
    val = ArchSequence(data[val_idx])

    # train parts are the left 6 slices
    train_parts = [p for i, p in enumerate(data)
                   if i not in (test_idx, val_idx)]
    assert len(train_parts) == 6

    # ArchSequence([list of all remain AudioSequence])
    train = ArchSequence([audio_files for slice in train_parts for audio_files in slice])

    # Generate the Callbacks
    verbose = 1
    m_loss = ModelCheckpoint(join(slice_dir, 'model_{epoch:03d}.h5'),
                             monitor='loss',
                             save_best_only=False)
    m_loss_best = ModelCheckpoint(join(slice_dir, 'model_best.h5'),
                                  monitor='loss',
                                  save_best_only=True)
    m_val_loss = ModelCheckpoint(join(slice_dir, 'model_best_val.h5'),
                                 monitor='val_loss',
                                 save_best_only=True)
    es = EarlyStopping(monitor='val_loss',
                       min_delta=1e-4,
                       patience=20,
                       verbose=verbose)
    tb = TensorBoard(log_dir=join(slice_dir, 'logs'),
                     write_graph=True,
                     write_images=True)
    callbacks = [m_loss, m_loss_best, m_val_loss, es, tb]

    # List of saved model name
    files = sorted(glob(join(slice_dir, 'model_???.h5')))

    # If train again with the the last exist train result
    if files:
        model_file = files[-1]
        initial_epoch = int(model_file[-6:-3])
        print('RESUMING TRAINING USING %s' % model_file)
        model_for_train = load_model(model_file)
    # If new train
    else:
        model_for_train = multi_model.model_NoDense()#model()
        initial_epoch = 0

    history = model_for_train.fit(train,
                                  steps_per_epoch=len(train),
                                  initial_epoch=initial_epoch,
                                  epochs=epochs,
                                  shuffle=True,
                                  validation_data=val,
                                  validation_steps=len(val),
                                  callbacks=callbacks
                                  )
    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.plot(history.history['loss'], label='test_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss at Slice' + str(test_idx))
    plt.ylabel('value')
    plt.xlabel('epochs')
    plt.legend(loc="upper left")
    plt.savefig(join(slice_dir, 'history_cache.png'))


def train(data, int_range, epochs):
    # i is the test_index
    for i in int_range:
        preparefortrain(data, i, epochs)

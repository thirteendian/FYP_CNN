import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *


def model_NoDense():
    """
    Front-end A
    Flatten
    Relu
    """
    m_NoDense = tf.keras.Sequential()
    m_NoDense.add(Conv2D(10, (7, 3), input_shape=(15, 80, 3),
                         padding='valid', activation='relu'))
    m_NoDense.add(MaxPooling2D(pool_size=(1, 3)))
    m_NoDense.add(Conv2D(20, (3, 3), input_shape=(9, 26, 10),
                         padding='valid', activation='relu'))
    m_NoDense.add(MaxPooling2D(pool_size=(1, 3)))
    m_NoDense.add(Dropout(0.5))
    m_NoDense.add(Flatten())
    m_NoDense.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.05, momentum=0.8, clipvalue=5)
    m_NoDense.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['binary_accuracy'])
    return m_NoDense



def model_NineLayers():
    """
    Front-end A
    back-end C
    Relu
    """
    m_NineLayers = tf.keras.Sequential()
    m_NineLayers.add(Conv2D(10, (7, 3), input_shape=(15, 80, 3),
                            padding='valid', activation='relu'))
    m_NineLayers.add(MaxPooling2D(pool_size=(1, 3)))
    m_NineLayers.add(Conv2D(20, (3, 3), input_shape=(9, 26, 10),
                            padding='valid', activation='relu'))
    m_NineLayers.add(MaxPooling2D(pool_size=(1, 3)))
    m_NineLayers.add(Dropout(0.5))
    ###########
    m_NineLayers.add(Conv2D(40, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(40, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(40, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(80, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(80, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(80, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Conv2D(135, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_NineLayers.add(BatchNormalization(axis=1))
    m_NineLayers.add(Flatten())
    m_NineLayers.add(Dropout(0.5))
    m_NineLayers.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.001, momentum=0.7, clipvalue=5)
    m_NineLayers.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['binary_accuracy'])
    return m_NineLayers


def model_FiveLayers():
    """
    Front-end A
    back-end D
    Relu
    """
    m_FiveLayers = tf.keras.Sequential()
    m_FiveLayers.add(Conv2D(10, (7, 3), input_shape=(15, 80, 3),
                            padding='valid', activation='relu'))
    m_FiveLayers.add(MaxPooling2D(pool_size=(1, 3)))
    m_FiveLayers.add(Conv2D(20, (3, 3), input_shape=(9, 26, 10),
                            padding='valid', activation='relu'))
    m_FiveLayers.add(MaxPooling2D(pool_size=(1, 3)))
    m_FiveLayers.add(Dropout(0.5))
    ######
    m_FiveLayers.add(Conv2D(60, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_FiveLayers.add(BatchNormalization(axis=1))
    m_FiveLayers.add(Conv2D(60, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_FiveLayers.add(BatchNormalization(axis=1))
    m_FiveLayers.add(Conv2D(60, (3, 3), padding='same',
                            data_format='channels_last', activation='relu'))
    m_FiveLayers.add(BatchNormalization(axis=1))
    m_FiveLayers.add(Flatten())
    m_FiveLayers.add(Dropout(0.5))
    m_FiveLayers.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=0.05, momentum=0.8, clipvalue=5)
    m_FiveLayers.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['binary_accuracy'])
    return m_FiveLayers
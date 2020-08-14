#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: parnian 
"""

from __future__ import print_function
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

K.set_image_data_format('channels_last')

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)

def tf_count(t, val):
    elements_equal_to_value = K.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def testLoss(y_true,y_pred):
    return K.sum(y_pred-y_true,axis=-1)

def binaryCrossEntropy(y_true, y_pred):
    return K.binary_crossentropy(y_true,y_pred)

class Capsule(Layer):
   

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,
        
       
           
        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)




batch_size = 65
num_classes = 2
epochs = 100

x_train= np.load(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\data\trainNPYFileFinal2ClassOneOutput\data.npy")
y_train= np.load(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\data\trainNPYFileFinal2ClassOneOutput\labels.npy")

x_valid=  np.load(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\data\testNPYFileFinal2ClassOneOutput\data.npy")
x_valid2 = tf.cast(x_valid, tf.float32)
y_valid=  np.load(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\data\testNPYFileFinal2ClassOneOutput\labels.npy")
y_valid2=  np.load(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\data\testNPYFileFinal2ClassOneOutput\labels.npy")

# here we handle class imbalance - factorCovid is detemening how "hard" we penalize for the mistake in covid classification
factorCovid = 4

nonCovidTrainSamples = np.count_nonzero(y_train==0)
covidTrainSamples = np.count_nonzero(y_train==1)
updatedCovidTrainSamplesForStrongerPenalty = covidTrainSamples/factorCovid

demod = updatedCovidTrainSamplesForStrongerPenalty+nonCovidTrainSamples

class_weights = {0: 1-nonCovidTrainSamples/demod,
                1: 1-updatedCovidTrainSamplesForStrongerPenalty/demod}

y_train = utils.to_categorical(y_train, num_classes)
y_valid = utils.to_categorical(y_valid, num_classes)



input_image = Input(shape=(None, None, 3))
x = Conv2D(64, (3, 3), activation='relu',trainable = False)(input_image)
x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(64, (3, 3), activation='relu',trainable = False)(x)
x = AveragePooling2D((2, 2),trainable = False)(x)
x = Conv2D(128, (3, 3), activation='relu',trainable = False)(x)
x = Conv2D(128, (3, 3), activation='relu',trainable = False)(x)




x = Reshape((-1, 128))(x)
x = Capsule(32, 8, 3, True)(x)  
x = Capsule(32, 8, 3, True)(x)

capsule = Capsule(5, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)




model = Model(inputs=[input_image], outputs=[output])

model.load_weights(r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\Models\2_pre-train_TrainOnAllDataSetsOurs.h5")

capsule2 = Capsule(2, 16, 3, True)(model.layers[-3].output)
# This is a change from the basic NN topology
x = Capsule(2, 64, 32, True)(capsule2)
# End of change
output2 = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(x)


model2 = Model(inputs=[input_image], outputs=[output2])
adam = optimizers.Adam(lr=0.001)
model2.compile(loss=margin_loss, optimizer=adam, metrics=['accuracy'])
model2.summary()

data_augmentation = False

# The best model is selected based on the loss value on the validation set

filepath="2_DifferentTopology-improvement-binary-after-{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# this callback allows to print out metrics during training
class myCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        return
    def on_train_end(self, logs=None):
        return
    def on_epoch_begin(self, epoch, logs=None):
        return
    def on_test_begin(self, logs=None):
        return
    def on_test_end(self, logs=None):
        return
    def on_predict_begin(self, logs=None):
        return
    def on_predict_end(self, logs=None):
        return
    def on_train_batch_begin(self, batch, logs=None):
        return
    def on_train_batch_end(self, batch, logs=None):
        return
    def on_test_batch_begin(self, batch, logs=None):
        return
    def on_test_batch_end(self, batch, logs=None):
        return
    def on_predict_batch_begin(self, batch, logs=None):
        return
    def on_predict_batch_end(self, batch, logs=None):
        return
    def on_epoch_end(self, epoch, logs=None):
        logForPlotPath = r"D:\Data\covid-caps-backup\GIT\tmp\covid-caps-project\2_logForPlotDifferentTopology.txt"
        logForPlotObj = open(logForPlotPath, "a")
        preds = self.model.predict([x_valid2])
        predIdx = np.argmax(preds, axis=1)
        equalityTensor = tf.equal(predIdx, y_valid2)
        accuracy = tf.reduce_sum(tf.cast(equalityTensor, tf.float32)) / len(equalityTensor)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(predIdx)):
            if predIdx[i] == 0 and y_valid2[i] == 0:
                TN += 1
            if predIdx[i] == 0 and y_valid2[i] == 1:
                FN += 1
            if predIdx[i] == 1 and y_valid2[i] == 0:
                FP += 1
            if predIdx[i] == 1 and y_valid2[i] == 1:
                TP += 1
        returningString = "Epoch=" + str(epoch)
        returningString += " Accuracy=" +str(round(100 * float(accuracy), 2))
        returningString += " TP=" + str(TP)
        returningString += " FP=" + str(FP)
        returningString += " TN=" + str(TN)
        returningString += " FN=" + str(FN)
        logForPlotObj.write(returningString+"\n")
        logForPlotObj.close()

callbacks_list = [checkpoint,myCallback()]





if not data_augmentation:
    print('Not using data augmentation.')
    model2.fit(
        [x_train], [y_train],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=[[x_valid], [y_valid]], class_weight=class_weights,
        shuffle=True,callbacks=callbacks_list)
    
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0.1,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        brightness_range=[0.5,1.5],
        shear_range=0.1,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.,  # set range for random chann el shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input 
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_valid, y_valid),shuffle=True)

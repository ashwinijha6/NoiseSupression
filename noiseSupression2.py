#In[0]:

# Importing library

# System
import os 
from os.path import isdir, join 
from pathlib import Path 
import pandas as pd # Preprocessing data

# Math
import numpy as np
from scipy.fftpack import fft 
from scipy import signal 
from scipy.io import wavfile 
import librosa 

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

# Import AI library
from keras.utils import Sequence, to_categorical
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Activation, Dense, Convolution2D, GlobalMaxPool2D
from keras.layers import LSTM, GRU, SimpleRNN, Dropout, concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
from keras import backend as K
from keras.constraints import Constraint
# In[1]:

# Build property model

class Config(object):
  def __init__(self, 
                sample_rate = 48000,
                lr = 0.001, 
                n_mfccs = 42, 
                n_classes = 2, 
                max_epochs = 10, 
                n_fft = 2,
                windowSize = 8):
    self.sample_rate = sample_rate
    self.lr = lr
    self.n_mfccs = n_mfccs
    self.n_classes = n_classes
    self.max_epochs = max_epochs
    self.n_fft = n_fft
    self.windowSize = windowSize
  
# In[2]: 

# Build data generator

class DataGenerator(Sequence):
  def __init__(self, 
                config,
                input_dir,
                target_dir,
                labels = None,
                batch_size = 64):
    self.config = config
    self.input_dir = input_dir
    self.target_dir = target_dir
    self.labels = labels
    self.batch_size = batch_size

    # Initialize input data for training
    self.input_data = []

    # Loop through all sound files in the input sound directory
    for aFile in os.listdir (self.input_dir):
      
      # Extracting sample rate and actual data as 1D array
      sample_rate, samples = wavfile.read(os.path.join(self.input_dir, aFile))
      
      # Make sure the sample rate is the same as the specified sample rate in config
      assert (sample_rate == self.config.sample_rate), " Audio file must be sampled at %d Hz"%self.config.sample_rate
      
      # Convert the actual data to MFCC delta data as 2D array
      samples = samples.astype('float')
      S = librosa.feature.melspectrogram(np.array(samples, dtype= float), sr=self.config.sample_rate, n_mels=100)
      log_S = librosa.power_to_db(S, ref=np.max)
      audio_mfcc = librosa.feature.mfcc(S=log_S,sr=self.config.sample_rate,n_mfcc=self.config.n_mfccs)
      delta2_mfcc = librosa.feature.delta(audio_mfcc, order=2)

      # Normalize to a int number of sequences of width windowSize
      nb_sequences = int (np.floor(len (delta2_mfcc[0]) / config.windowSize))
      new_delta = delta2_mfcc[:,:nb_sequences * self.config.windowSize]
      
      # Reshape data to seperated nb_sequnces number of frames with width self.config.windowSize and height self.config.n_mfccs
      new_delta = np.reshape(new_delta, (nb_sequences, self.config.n_mfccs, self.config.windowSize))

      # If the array is empty, then append the first sequences of elements
      if len(self.input_data) == 0:
        self.input_data = new_delta
      else:
        # Else, concatenate those sequences to the data
        self.input_data = np.concatenate((self.input_data, new_delta), axis = 0)
        print("Shape of current input_delta data: %s"%str(new_delta.shape))
        print("Shape of current input data: %s"%str(np.asarray(self.input_data).shape))

    # Initialize input data for training
    self.target_data = []
    
    # Loop through all sound files in the input sound directory
    for aFile in os.listdir (self.target_dir):
      
      # Extracting sample rate and actual data as 1D array
      sample_rate, samples = wavfile.read(os.path.join(self.target_dir, aFile))
      
      # Make sure the sample rate is the same as the specified sample rate in config
      assert (sample_rate == self.config.sample_rate), " Audio file must be sampled at %d Hz"%self.config.sample_rate
      
      # Convert the actual data to MFCC delta data as 2D array
      samples = samples.astype('float')
      S = librosa.feature.melspectrogram(np.array(samples, dtype= float), sr=self.config.sample_rate, n_mels=100)
      log_S = librosa.power_to_db(S, ref=np.max)
      audio_mfcc = librosa.feature.mfcc(S=log_S,sr=self.config.sample_rate,n_mfcc=self.config.n_mfccs)
      delta2_mfcc = librosa.feature.delta(audio_mfcc, order=2)

      # Normalize to a int number of sequences of width windowSize
      nb_sequences = int (np.floor(len (delta2_mfcc[0]) / config.windowSize))
      new_delta = delta2_mfcc[:,:nb_sequences * self.config.windowSize]
      
      # Reshape data to seperated nb_sequnces number of frames with width self.config.windowSize and height self.config.n_mfccs
      new_delta = np.reshape(new_delta, (nb_sequences, self.config.n_mfccs, self.config.windowSize))

      # If the array is empty, then append the first sequences of elements
      if len(self.target_data) == 0:
        self.target_data = new_delta
      else:
        # Else, concatenate those sequences to the data
        self.target_data = np.concatenate((self.target_data, new_delta), axis = 0)
        print("Shape of target_delta data: %s"%str(new_delta.shape))
        print("Shape of current target data: %s"%str(np.asarray(self.target_data).shape))

    self.on_epoch_end()

    print(self.input_data)
  
  def __len__(self):
    # Return the number of batches to be trained
    return int(np.ceil(len(self.input_data) / self.batch_size))
    # return len(self.input_data)

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.input_data))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    input_batch_temp = [self.input_data[k] for k in indexes]
    
    target_batch_temp = [self.target_data[k] for k in indexes]
    return self.__data_batch_generation(input_batch_temp, target_batch_temp)

  def __data_batch_generation(self, input_batch_temp, target_batch_temp):
    cur_batch_size = len(input_batch_temp)

    # print("Shape of input batch: %d"%)
    
    X = np.zeros((cur_batch_size, self.config.n_mfccs, self.config.windowSize))
    Y = np.zeros((cur_batch_size, self.config.n_mfccs, self.config.windowSize))

    # X = np.zeros((self.dim[0], len(input_batch_temp[0][0])))
    # Y = np.zeros((self.dim[0], len(target_batch_temp[0][0])))

    print("Initializing X: %s"%str(X.shape))
    for i, datum in enumerate(input_batch_temp):

      print("each audio data: %s"%str(datum.shape))

      # X[i,] = np.expand_dims(datum, -1)
      X[i,] = datum
      

    print ("post initializing X: %s"%str(X.shape))

    for i, datum in enumerate(target_batch_temp):
      # Y[i,] = np.expand_dims(datum, -1)
      Y[i,] = datum

    return (X, Y)

#In[3]:
  
# Actual Work


clean_sound_path_folder = './data/clean_trainset_28spk_wav/clean_trainset_28spk_wav/'
noisy_sound_path_folder = './data/noisy_trainset_28spk_wav/noisy_trainset_28spk_wav/'
test_file_path = os.path.join(clean_sound_path_folder, 'p226_001.wav')
sample_rate, samples = wavfile.read(test_file_path)

audio_length = len(samples) / float(sample_rate)

config = Config(sample_rate=sample_rate,
                max_epochs=1,)

# Test

samples = samples.astype('float')
S = librosa.feature.melspectrogram(np.array(samples, dtype= float), sr=config.sample_rate, n_mels=100)
log_S = librosa.power_to_db(S, ref=np.max)
audio_mfcc = librosa.feature.mfcc(S=log_S,sr=config.sample_rate,n_mfcc=config.n_mfccs)
delta2_mfcc = librosa.feature.delta(audio_mfcc, order=2)

ipd.Audio(samples, rate= sample_rate)

#In[4]:

# Generate data

train_generator = DataGenerator(config, noisy_sound_path_folder, clean_sound_path_folder)

validation_generator = DataGenerator(config, noisy_sound_path_folder, clean_sound_path_folder)

#In[5]: 

# Build model:
def get_2d_dummy_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.n_mfccs, config.windowSize, 1))
    x = Flatten()(inp)
    x = Dense(32, activation='relu')(inp)
    out = Dense(nclass, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(config.lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model


def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)
reg = 0.000001
constraint = WeightClip(0.499)
def đâyLàMôĐồRờNờNờ (config) :
  main_input = Input(shape=(None, 42), name='main_input')
  tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
  vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
  vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
  noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
  noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
  denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

  denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

  denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

  model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

  model.compile(loss=[mycost, my_crossentropy],
                metrics=[msse],
                optimizer='adam', loss_weights=[10, 0.5])

def get_2d_conv_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.windowSize,config.n_mfccs,1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(config.lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model


#In[6]:

model = đâyLàMôĐồRờNờNờ (config)
model.summary()

#In[7]:

history = model.fit_generator(generator = train_generator,
                              epochs = 1,
                              verbose = 1,
                              validation_data = validation_generator)

#In[8]:

# history = model(x = train_generator.__getitem__(3)[0],
#                 y = train_generator.__getitem__(3)[1],
#                               epochs = 1,
#                               batch_size = 2,
#                               verbose = 1)

#%%

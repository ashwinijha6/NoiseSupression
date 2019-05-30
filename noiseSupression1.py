#In[0]:

import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
from DataUtils import load_wav_files

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import librosa

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

%matplotlib inline

# Import AI library
from keras.models import Model
from keras.layers import Reshape, Dense, LSTM, Conv1D, Conv2D, Dense, RNN, GRU, Input, Activation, MaxPooling1D, MaxPooling2D, Flatten, Convolution2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, UpSampling1D
from keras.optimizers import Adam, Adadelta
from keras.utils import Sequence, to_categorical
from keras import backend as K
from keras.utils import Sequence

from keras.preprocessing.image import ImageDataGenerator

#In[]:

# Test Mode
# Change this to true to see the result

COMPLETE_RUN = False

#In[]:
class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=8, n_classes=42,
                 use_mfcc=True, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=22, n_fft = 1):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


#In[1]:

train_audio_path = './data/dataset/'
clean_speech_test = 'clean_trainset_28spk_wav/clean_trainset_28spk_wav/p226_008.wav'
noisy_speech_test = 'noisy_trainset_28spk_wav/noisy_trainset_28spk_wav/p226_008.wav'
sample_rate_clean, samples_clean = wavfile.read(str(train_audio_path) + clean_speech_test)
sample_rate_noisy, samples_noisy = wavfile.read(str(train_audio_path) + noisy_speech_test)

#In[]:

print('clean:')
print(sample_rate_noisy)
ipd.Audio(samples_clean, rate=sample_rate_clean)

#In[]:

print('noisy:')
print(sample_rate_noisy)
ipd.Audio(samples_noisy, rate=sample_rate_noisy)

#In[2]:

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

#In[3]:

# Clean sound plot

freqs, times, spectrogram = log_specgram(samples_clean, sample_rate_clean)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + clean_speech_test)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(samples_clean) / sample_rate_clean, len(samples_clean)), samples_clean)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + clean_speech_test)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

#Listen
ipd.Audio(samples_clean, rate = sample_rate_clean)


#In[3]:

# Noisy sound plot

freqs, times, spectrogram = log_specgram(samples_noisy, sample_rate_noisy)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + noisy_speech_test)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(samples_noisy) / sample_rate_noisy, len(samples_noisy)), samples_noisy)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + noisy_speech_test)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

#Listen
ipd.Audio(samples_noisy, rate = sample_rate_noisy)

#In[4]:

# From this tutorial
# https://github.com/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
S_noisy = librosa.feature.melspectrogram(np.array(samples_noisy, dtype= float), sr=sample_rate_noisy, n_mels=100)
S_clean = librosa.feature.melspectrogram(np.array(samples_clean, dtype= float), sr=sample_rate_clean, n_mels=100)
# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S_noisy = librosa.power_to_db(S_noisy, ref=np.max)
log_S_clean = librosa.power_to_db(S_clean, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sample_rate_noisy, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram ')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

#In[5]:

mfcc_clean = librosa.feature.mfcc(S=log_S_clean, n_mfcc=22)
mfcc_noisy = librosa.feature.mfcc(S=log_S_noisy, n_mfcc=22)

# Let's pad on the first and second deltas while we're at it
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()

#In[6]:

def rnn_model (input_shape = (22, 809, 1)):
    input_neuron = Input(input_shape)

    neuron_layer = Conv2D (filters=32, kernel_size=(3,3), activation='relu') (input_neuron)
    neuron_layer = MaxPooling2D(pool_size = (2,)) (neuron_layer)
    neuron_layer = Flatten() (neuron_layer)

    # neuron_layer = LSTM (32)(neuron_layer)

    neuron_layer = Dense (32, activation = 'relu') (neuron_layer)

    neuron_layer = Dense(2, kernel_initializer='he_normal',
                  name='dense')(neuron_layer)
    pred = Activation('softmax', name='softmax') (neuron_layer)

    model = Model(input_neuron, pred)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

#In[]:

def get_2d_dummy_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = GlobalMaxPool2D()(inp)
    out = Dense(nclass, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model

def get_2d_conv_model(config):
    
    nclass = config.n_classes
    
    inp = Input(shape=(config.dim[0],config.dim[1],1))
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
    opt = Adam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model

#In[8]:

def deep_1d_autoencoder():

	# ENCODER
	input_sig = Input(batch_shape=(None,800,1))
	x = Conv1D(256,32, activation='relu', padding='same')(input_sig)
	x1 = MaxPooling1D(2)(x)
	x2 = Conv1D(256,32, activation='relu', padding='same')(x1)
	x3 = MaxPooling1D(2)(x2)
	x4 = Conv1D(128,16, activation='relu', padding='same')(x3)
	x5 = MaxPooling1D(2)(x4)
	x6 = Conv1D(128,16, activation='relu', padding='same')(x5)
	x7 = MaxPooling1D(2)(x6)
	x8 = Conv1D(64,8, activation='relu', padding='same')(x7)
	x9 = MaxPooling1D(2)(x8)
	flat = Flatten()(x9)
	encoded = Dense(32,activation = 'relu')(flat)
    
	x8_ = Conv1D(64, 8, activation='relu', padding='same')(x9)
	x7_ = UpSampling1D(2)(x8_)
	x6_ = Conv1D(128, 16, activation='relu', padding='same')(x7_)
	x5_ = UpSampling1D(2)(x6_)
	x4_ = Conv1D(128, 16, activation='relu', padding='same')(x5_)
	x3_ = UpSampling1D(2)(x4_)
	x2_ = Conv1D(256, 32, activation='relu', padding='same')(x3_)
	x1_ = UpSampling1D(2)(x2_)
	x_ = Conv1D(256, 32, activation='relu', padding='same')(x1_)
	upsamp = UpSampling1D(2)(x_)
	flat = Flatten()(upsamp)
	decoded = Dense(800,activation = 'relu')(flat)
	decoded = Reshape((800,1))(decoded)
	
	#print("shape of decoded {}".format(keras.int_shape(decoded)))

	return input_sig, decoded


#In[8]:

# Prepare config

config = Config(sampling_rate=48000, audio_duration=8, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=22)
if not COMPLETE_RUN:
    config = Config(sampling_rate=48000, audio_duration=8, n_folds=2, 
                    max_epochs=1, use_mfcc=True, n_mfcc=22)


#In[9]:

# Prepare data

def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X

#In[9]:

class AudioDataGenerator:
    def __init__(self, input_data_path, target_data_path, nb_mfccs=26, nb_fft_point=2,batch_size=32, sample_rates=16000, bit_sizes=16, channel_numbers=1, shuffle=True, duration_in_ms=30, overlap=10):
        self.input_data_path=input_data_path
        self.target_data_path=target_data_path
        self.batch_size=batch_size
        self.sample_rates=sample_rates
        self.bit_sizes=bit_sizes
        self.channel_numbers=channel_numbers
        self.shuffle=shuffle
        self.duration_in_ms=duration_in_ms
        self.overlap=overlap
        self.samples = []
        self.nb_mfccs=nb_mfccs
        self.nb_fft_point=nb_fft_point

        self.cur_index = 0
        self.is_build=None

    @staticmethod
    def log_specgram(audio, sample_rate, window_size =30,step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        #print((window_size,step_size), (nperseg,noverlap),sample_rate)
        freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)        
    
    def build_data_mfcc(self):
            #Build the feature using mfcc transformation
        self.is_build=True
        self.w=self.nb_mfccs*self.nb_fft_point#int(round(self.sample_rates*self.duration_in_ms/1e3)/2 +1) #Number of frequency number = number of times sample/2 +1
        self.d=1
        self.audio_data= []
        for sample in self.samples:
            y_label = sample[0]
            filename = sample[1]
            print('Reading file: ',filename)
            sample_rate, samples = wavfile.read(filename)
            assert sample_rate==self.sample_rates, " Audio file must be sampled at %d Hz"%self.sample_rates
            samples = samples.astype('float')
            #Calculate the mffc of each frame
            
            audio_mfcc = librosa.feature.mfcc(samples,sr=sample_rate,n_mfcc=self.nb_mfccs)
            
            for idx in range(int(round(audio_mfcc.shape[-1]/self.nb_fft_point)) -1):
                mfccs = audio_mfcc[:,idx*self.nb_fft_point:(idx+1)*self.nb_fft_point]
                mfccs = mfccs.reshape(mfccs.shape[0]*mfccs.shape[1],1)
                self.audio_data.append([y_label,mfccs])
                #print(mfccs.shape)
                #print( start_sample_in_frame_idx,' ', len(samples))
                
            
        self.n = len(self.audio_data)
        self.indexes = list(range(self.n))
        print('Number of sample: ', self.n)
    
    def number_sample(self):
        return self.n
    #Get the next sample
    def _next_sample(self):
        if self.cur_index >= self.n:
            self.cur_index=0
        if self.shuffle and self.cur_index == 0:
            random.shuffle(self.indexes)
        #Return current index
        y_label = self.audio_data[self.indexes[self.cur_index]][0]
        spec = self.audio_data[self.indexes[self.cur_index]][1]
        self.cur_index = self.cur_index +1
        return y_label, spec
    def flow_data(self):
        assert self.is_build==True," You must build the data before flow it"
        while True:
            x_data = np.zeros([self.batch_size,self.w,self.d])
            y_data = np.zeros([self.batch_size,len(self.classes)])
            for batch in range(self.batch_size):
                y_label,spec= self._next_sample()
                x_data[batch]=spec
                y_data[batch][self.classes.index(y_label)] = 1
            yield (x_data, y_data)

#In[9]:

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, input_data, target_data, dim, n_channels=1, shuffle=True, duration = 8):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.input_data = input_data
        self.target_data = target_data
        self.duration = duration
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        self.waa=int(round(48000*self.duration)/4 +1) #Number of frequency number = number of times sample/2 +1
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.input_data[ID]

            # Store target
            y[i,] = self.target_data[ID]

        yield X, y


#In[10]:

model_input, model_target = deep_1d_autoencoder()
model = Model(inputs=model_input, outputs= model_target)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['acc'])
model.summary()

#In[11]:

# Build data



batch_size = 32
# train_generator = AudioDataGenerator(train_data_folder,
#                                     batch_size = 20,
#                                     nb_mfccs= config.n_mfcc,
#                                     sample_rates = config.sampling_rate,
#                                     duration_in_ms = 8,
#                                     overlap = 20)

# validation_generator = AudioDataGenerator(validation_data_folder,
#                                     batch_size = 20,
#                                     nb_mfccs= config.n_mfcc,
#                                     sample_rates = config.sampling_rate,
#                                     duration_in_ms = 8,
#                                     overlap = 20)

# train_generator.build_data_mfcc()
# validation_generator.build_data_mfcc()

input_dir = "./data/dataset/noisy_trainset_28spk_wav/noisy_trainset_28spk_wav/"
target_dir = "./data/dataset/clean_trainset_28spk_wav/clean_trainset_28spk_wav/"

input_Aug = load_wav_files(input_dir)
target_Aug = load_wav_files(target_dir)

input_data = []
input_indices = []
target_data = []
target_indices = []

for i, data in enumerate(input_Aug):
    input_data.append(data.data)
    input_indices.append(i)

for i, data in enumerate(target_Aug):
    target_data.append(data.data)
    target_indices.append(i)

input_np_data = np.asarray([x for x in input_data])
target_np_data = np.asarray([x for x in target_data])

# input_np_data = np.resize(input_np_data, (len(input_np_data),800,1))
# target_np_data = np.resize(target_np_data, (len(target_np_data),800,1))

# dim = (800, )

#In[10]:

training_generator = DataGenerator(list_IDs = target_indices,
                                    batch_size = batch_size,
                                    input_data = input_np_data,
                                    target_data = target_np_data,
                                    dim = dim,
                                    n_channels = 1,
                                    shuffle = True)
                                    
validation_generator = DataGenerator(list_IDs = target_indices,
                                        batch_size = batch_size,
                                        input_data = input_np_data,
                                        target_data = target_np_data,
                                        dim = dim,
                                        n_channels = 1,
                                        shuffle = True)


#In[12]:

# history = model.fit_generator(generator=training_generator,
#                     steps_per_epoch=1,
#                     epochs=1,
#                     verbose=1,
#                     validation_data= validation_generator,
#                     validation_steps=1)

input_np_data = np.reshape(input_np_data, (*input_np_data.shape, 1))
target_np_data = np.reshape(target_np_data, (*target_np_data.shape, 1))

input_datagen = ImageDataGenerator()
target_datagen = ImageDataGenerator()

# Provide the same seed and keyword arguments to the fit and flow methods   
seed = 1
input_datagen.fit(input_np_data, augment=False, seed=seed)
target_datagen.fit(target_np_data, augment=False, seed=seed)

input_generator = input_datagen.flow(input_np_data, None, batch_size=16, seed=seed)

target_generator = target_datagen.flow(target_np_data, None, batch_size=16, seed=seed)

train_generator = zip(input_generator, target_generator)

model.fit_generator(generator=train_generator,
                            validation_data=train_generator,
                            steps_per_epoch=input_np_data.shape[0]//16,
                            validation_steps=input_np_data.shape[0]//16,
                            shuffle=True,
                            epochs=1,
                            verbose=1)


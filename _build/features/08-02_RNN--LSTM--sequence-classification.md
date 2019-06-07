---
redirect_from:
  - "/features/08-02-rnn--lstm--sequence-classification"
interact_link: content/features/08-02_RNN--LSTM--sequence-classification.ipynb
kernel_name: conda-env-mujoco2-py
title: 'RNN Classification'
prev_page:
  url: /features/features
  title: 'RNN'
next_page:
  url: 
  title: ''
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# IMDB data

[link](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

sequence datasets
- [datasetsome](https://dataloaderx.github.io/datasetsome/), [online notebooks](https://mybinder.org/v2/gh/DataLoaderX/datasetsome/master)
- [google-quick-draw](https://github.com/googlecreativelab/quickdraw-dataset),  [dataset show online](https://quickdraw.withgoogle.com/data)



{:.input_area}
```python
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)
```




{:.input_area}
```python
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```


{:.output .output_stream}
```
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
17465344/17464789 [==============================] - 2s 0us/step

```

Pads sequences to the same length



{:.input_area}
```python
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_train[:5])
```


{:.output .output_stream}
```
[[   0    0    0 ...   19  178   32]
 [   0    0    0 ...   16  145   95]
 [   0    0    0 ...    7  129  113]
 [ 687   23    4 ...   21   64 2574]
 [   0    0    0 ...    7   61  113]]

```

define, compile and fit the model using keras



{:.input_area}
```python
embedding_vec_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vec_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
```


{:.output .output_stream}
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
=================================================================
Total params: 213,301
Trainable params: 213,301
Non-trainable params: 0
_________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/3
25000/25000 [==============================] - 170s 7ms/step - loss: 0.4647 - acc: 0.7802 - val_loss: 0.3658 - val_acc: 0.8430
Epoch 2/3
25000/25000 [==============================] - 180s 7ms/step - loss: 0.3036 - acc: 0.8766 - val_loss: 0.3259 - val_acc: 0.8666
Epoch 3/3
25000/25000 [==============================] - 178s 7ms/step - loss: 0.2617 - acc: 0.8962 - val_loss: 0.3250 - val_acc: 0.8648

```




{:.output .output_data_text}
```
<keras.callbacks.History at 0x7f7f6c7237f0>
```





{:.input_area}
```python
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```


{:.output .output_stream}
```
Accuracy: 86.48%

```

using dropout



{:.input_area}
```python
from keras.layers import Dropout
model1 = Sequential()
model1.add(Embedding(top_words, embedding_vec_length, input_length=max_review_length))
model1.add(Dropout(0.2))
model1.add(LSTM(100))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
scores1 = model1.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
```


using dropout in the LSTM, Keras provides this capability with parameters on the LSTM layer, the dropout for configuring the input dropout and recurrent_dropout for configuring the recurrent dropout.



{:.input_area}
```python
model2 = Sequential()
model2.add(Embedding(top_words, embedding_vec_length, input_length=max_review_length))
model2.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
scores2 = model2.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
```


{:.output .output_stream}
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
lstm_3 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 101       
=================================================================
Total params: 213,301
Trainable params: 213,301
Non-trainable params: 0
_________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/3
25000/25000 [==============================] - 195s 8ms/step - loss: 0.5017 - acc: 0.7523 - val_loss: 0.3863 - val_acc: 0.8303
Epoch 2/3
25000/25000 [==============================] - 201s 8ms/step - loss: 0.3676 - acc: 0.8471 - val_loss: 0.3621 - val_acc: 0.8477
Epoch 3/3
25000/25000 [==============================] - 199s 8ms/step - loss: 0.3382 - acc: 0.8623 - val_loss: 0.3517 - val_acc: 0.8515
Accuracy: 85.15%

```

using CNN with rnn to also capture the spatial structure.



{:.input_area}
```python
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model3 = Sequential()
model3.add(Embedding(top_words, embedding_vec_length, input_length=max_review_length))
model3.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(LSTM(100))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
model3.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores3 = model3.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
```


{:.output .output_stream}
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 500, 32)           3104      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 100)               53200     
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 101       
=================================================================
Total params: 216,405
Trainable params: 216,405
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/3
25000/25000 [==============================] - 74s 3ms/step - loss: 0.4327 - acc: 0.7894
Epoch 2/3
25000/25000 [==============================] - 84s 3ms/step - loss: 0.2488 - acc: 0.9018
Epoch 3/3
25000/25000 [==============================] - 80s 3ms/step - loss: 0.2043 - acc: 0.9242
Accuracy: 88.36%

```

# RNN - Human motion recognition

learning from [this post](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)

datasets is downloaded from [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/)

## Data

- number of features: 9
- window size: 128 (128 time steps of 9 features are considered as one input vector)
- total number of samples: 7352 (training), 2947 (testing)



{:.input_area}
```python
%matplotlib inline
import os
import numpy as np
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJ_DIR = '.'
DATASETS_DIR= 'datasets/RNN-human-motion-recognition/UCI_HAR_Dataset'

def load_file(filepath):
    dataframe = read_csv(filepath, hearder=None, delim_whitespace=True)
    return dataframe.values
```


## LSTM approach



{:.input_area}
```python
# lstm model
%matplotlib inline
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group: train, test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix='./datasets/RNN-human-motion-recognition/'):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI_HAR_Dataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI_HAR_Dataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
```




{:.input_area}
```python
# run the experiment
run_experiment()
```


{:.output .output_stream}
```
(7352, 128, 9) (7352, 1)
(2947, 128, 9) (2947, 1)
(7352, 128, 9) (7352, 6) (2947, 128, 9) (2947, 6)
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
>#1: 88.463
>#2: 91.211
>#3: 90.397
>#4: 85.341
>#5: 90.601
>#6: 90.024
>#7: 87.682
>#8: 90.804
>#9: 89.243
>#10: 84.662
[88.46284356973193, 91.21140142517815, 90.39701391245333, 85.34102477095351, 90.60061079063453, 90.02375296912113, 87.68238887003733, 90.80420766881574, 89.24329826942655, 84.66236851034951]
Accuracy: 88.843% (+/-2.185)

```

## CNN + LSTM

- split the 128-time-step window to 4 sub-sequences of 32-time-step 



{:.input_area}
```python
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
```




{:.input_area}
```python
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features))) 
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

```




{:.input_area}
```python
run_experiment()
```


{:.output .output_stream}
```
(7352, 128, 9) (7352, 1)
(2947, 128, 9) (2947, 1)
(7352, 128, 9) (7352, 6) (2947, 128, 9) (2947, 6)
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /home/jianfeng/anaconda3/envs/mujoco2/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
>#1: 89.922
>#2: 90.974
>#3: 90.601
>#4: 88.157
>#5: 90.024
>#6: 90.940
>#7: 90.567
>#8: 91.144
>#9: 90.295
>#10: 89.718
[89.92195453003053, 90.97387173396675, 90.60061079063453, 88.15744825246013, 90.02375296912113, 90.93993892093654, 90.56667797760434, 91.14353579911774, 90.29521547336275, 89.71835765184933]
Accuracy: 90.234% (+/-0.827)

```

## ConvLSTM

used for spatio-temporal data

in Keras the `ConvLSTM2D(samples, time, rows, cols, channels)` class 

- samples: total number of samples 7352 (training)
- time: 4, the number of sub-sequences
- rows: 1, here we have 1-dim sequence data
- cols: 32, 32 time steps per sub-sequence. Think of rows x cols as an image
- channels: 9, for 9 features





{:.input_area}
```python
from keras.layers import ConvLSTM2D

def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size = 0, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy
```




{:.input_area}
```python
run_experiment()
```


{:.output .output_stream}
```
(7352, 128, 9) (7352, 1)
(2947, 128, 9) (2947, 1)
(7352, 128, 9) (7352, 6) (2947, 128, 9) (2947, 6)
>#1: 89.141
>#2: 89.922
>#3: 92.094
>#4: 90.363
>#5: 91.177
>#6: 91.992
>#7: 90.736
>#8: 91.347
>#9: 88.836
>#10: 90.770
[89.14149983033593, 89.92195453003053, 92.09365456396336, 90.36308109942314, 91.17746861214795, 91.99185612487275, 90.73634204275535, 91.34713267729894, 88.83610451407542, 90.77027485578554]
Accuracy: 90.638% (+/-1.042)

```

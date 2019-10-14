'''
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, BatchNormalization, Conv1D, GRU, TimeDistributed, Activation, Dense, Flatten
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint   
from keras.losses import categorical_crossentropy
'''
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, BatchNormalization, Conv1D, Conv2D, GRU, TimeDistributed, Activation, Dense, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint   
from tensorflow.keras.losses import categorical_crossentropy

from utils.model_utils import *
import os
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
def speech_to_text_model(input_dim, 
                         filters, 
                         kernel_size, 
                         strides,
                         padding, 
                         rnn_units, 
                         output_dim=29,
                         cell=GRU, 
                         activation='relu'):
    """ 
    Creates simple Conv-RNN model used for speech_to_text approach.

    :params:
    	input_dim - Integer, size of inputs (Example: 161 if using spectrogram, 13 for mfcc)
    	filters - Integer, number of filters for the Conv1D layer
		kernel_size - Integer, size of kernel for Conv layer
		strides - Integer, stride size for the Conv layer
		padding - String, padding version for the Conv layer ('valid' or 'same')
		rnn_units - Integer, number of units/neurons for the RNN layer(s)
		output_dim - Integer, number of output neurons/units at the output layer
							  NOTE: For speech_to_text approach, this number will be number of characters that may occur
		cell - Keras function, for a type of RNN layer * Valid solutions: LSTM, GRU, BasicRNN
		activation - String, activation type at the RNN layer

	:returns:
		model - Keras Model object

    """

    #Defines Input layer for the model
    input_data = Input(name='inputs', shape=(None, input_dim))

	#Defines 1D Conv block (Conv layer +  batch norm)
    conv_1d = Conv1D(filters, 
                     kernel_size, 
                     strides=strides, 
                     padding=padding,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    #Defines block (RNN layer + batch norm)
    simp_rnn = cell(rnn_units, 
                   activation=activation,
                   return_sequences=True, 
                   implementation=2, 
                   name='rnn')(bn_cnn)

    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)

    #Apply Dense layer to each time step of the RNN with TimeDistributed function
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    #Define model predictions with softmax activation
    y_pred = Activation('softmax', name='softmax')(time_dense)

    #Defines Model itself, and use lambda function to define output length based on inputs
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, padding, strides)
    print(model.summary())
    return model


def classification_model(input_dim, 
						 filters, 
						 kernel_size, 
						 strides,
    					 padding, 
    					 rnn_units=128,#256 
    					 output_dim=30, 
    					 dropout_rate=0.5, 
    					 cell=GRU, 
    					 activation='tanh'):
    """ 
    Creates simple Conv-Bi-RNN model used for word classification approach.

    :params:
    	input_dim - Integer, size of inputs (Example: 161 if using spectrogram, 13 for mfcc)
    	filters - Integer, number of filters for the Conv1D layer
		kernel_size - Integer, size of kernel for Conv layer
		strides - Integer, stride size for the Conv layer
		padding - String, padding version for the Conv layer ('valid' or 'same')
		rnn_units - Integer, number of units/neurons for the RNN layer(s)
		output_dim - Integer, number of output neurons/units at the output layer
							  NOTE: For speech_to_text approach, this number will be number of characters that may occur
		dropout_rate - Float, percentage of dropout regularization at each RNN layer, between 0 and 1
		cell - Keras function, for a type of RNN layer * Valid solutions: LSTM, GRU, BasicRNN
		activation - String, activation type at the RNN layer

	:returns:
		model - Keras Model object

    """

    #Defines Input layer for the model
    input_data = Input(name='inputs', shape=input_dim)

    #Defines 1D Conv block (Conv layer +  batch norm)
    conv_1d = Conv1D(filters, 
    		    kernel_size, 
                     strides=strides, 
                     padding=padding,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)

    #Defines Bi-Directional RNN block (Bi-RNN layer + batch norm)
    layer = cell(rnn_units, activation=activation,
                return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
    layer = BatchNormalization(name='bt_rnn_1')(layer)

    #Defines Bi-Directional RNN block (Bi-RNN layer + batch norm)
    #layer = cell(rnn_units, activation=activation,
    #            return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
    #layer = BatchNormalization(name='bt_rnn_final')(layer)
    
    layer = Flatten()(layer)

    #squish RNN features to match number of classes
    time_dense = Dense(output_dim)(layer)

    #Define model predictions with softmax activation
    y_pred = Activation('softmax', name='softmax')(time_dense)

    #Defines Model itself, and use lambda function to define output length based on inputs
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, padding, strides)

    print(model.summary())
    return model

def createNaiveModel(input_dim, strides, output_dim):
    input_data = Input(name = 'input', shape = input_dim)

    #output = Conv2D(name = 'conv2d_1', filters = 128, kernel_size = 3, strides = strides, padding = 'valid')(input_data)
    #output = Conv2D(name = 'conv2d_2', filters = 64, kernel_size = 3, strides = strides, padding = 'valid')(output)
    #output = Conv2D(name = 'conv2d_3', filters = 32, kernel_size = 3, strides = strides, padding = 'valid')(output)
    output = Flatten()(input_data)
    output = Dense(320)(output)
    output = Dense(160)(output)
    output = Dense(64)(output)
    output = Dense(output_dim)(output)
    output = Activation('softmax', name = 'softmax')(output)
    model = Model(inputs=input_data, outputs=output)
    print(model.summary())
    return model


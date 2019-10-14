# Main file to train the speech commands recognition model
# 0.preprocess
# 1.load data into required dataset format
# 2.compile the model, and suit to tff
# 3.train and test

from constant import *
from utils.dataset_utils import load_dataset
from utils.model_utils import add_categorical_loss
from models import classification_model,createNaiveModel
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import os, collections

tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # restrain the unneedd tensorflow output

dataset_train = load_dataset(DATASET_FILENAME)
#Expect tf.data.Dataset returned
print(len(dataset_train.client_ids), dataset_train.output_types, dataset_train.output_shapes)

example_dataset = dataset_train.create_tf_dataset_for_client(dataset_train.client_ids[0])
example_element = iter(example_dataset).next()

print(example_element['label'].numpy())
print(example_element['pixels'].numpy().shape)

def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([('x', tf.reshape(element['pixels'], [99, 161, 1])),
            ('y', tf.reshape(element['label'], [1])),])
    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(
            lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

sample_clients = dataset_train.client_ids[0 : NUM_CLIENTS] # 10 clients
federated_train_data = make_federated_data(dataset_train, sample_clients)
print(len(federated_train_data), federated_train_data[0])

def create_compiled_keras_model():
    optimizer = SGD(lr = 0.02, decay = 1e-6, momentum = 0.9, nesterov = True, clipnorm = 5) 
    '''
    model = classification_model(input_dim = (99, 161),
                                filters = 256, 
                                kernel_size = 1,
                                strides = 1,
                                padding = 'valid',
                                output_dim = 4)
    '''
    model = createNaiveModel(input_dim = ( 99, 161, 1), strides = 2, output_dim = 4)
    #model = add_categorical_loss(model, 4)
    # the keras classification model 
                  
    #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    #model.compile(loss = {'categorical_crossentropy' : lambda y_true, y_pred : y_pred}, optimizer = optimizer)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = [metrics.CategoricalAccuracy()])
    return model


def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


iterative_process = tff.learning.build_federated_averaging_process(model_fn)
print(str(iterative_process.initialize.type_signature))

state = iterative_process.initialize()
for round_num in range(2, 11):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

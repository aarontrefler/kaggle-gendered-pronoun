"""Utility functions for Keras modeling""" 
from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko


def build_mlp_model(input_shape, dense_layer_sizes, dropout_rate, lambd):
    """Taken from public Kaggle kernal: Taming the BERT - a baseline"""
    X_input = layers.Input(input_shape)

    # First dense layer
    X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
    X = layers.BatchNormalization(name = 'bn0')(X)
    X = layers.Activation('relu')(X)
    X = layers.Dropout(dropout_rate, seed = 7)(X)

    # Second dense layer
#   X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)
#   X = layers.BatchNormalization(name = 'bn1')(X)
#   X = layers.Activation('relu')(X)
#   X = layers.Dropout(dropout_rate, seed = 9)(X)

    # Output layer
    X = layers.Dense(3, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
    X = layers.Activation('softmax')(X)

    # Create model
    model = models.Model(input = X_input, output = X, name = "classif_model")
    return model

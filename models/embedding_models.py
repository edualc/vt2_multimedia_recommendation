import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.client import device_lib

def gpu_available():
    return tf.test.is_gpu_available()

def keyframe_embedding_model__bilstm(n_classes=13606, n_genres=20, input_shape=(224,224,3), sequence_length=20, \
    rating_head=True, genre_head=True, class_head=True, self_supervised_head=True, \
    learning_rate=3e-4, intermediate_activation='elu', l2_beta=0.01):
    
    keras_loss_functions = dict()
    keras_metrics = dict()
    keras_loss_weights = dict()
    network_heads = list()

    mobilenet = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    mobilenet.trainable = False

    input_layer = keras.layers.Input(shape=(sequence_length,) + input_shape)
    features = keras.layers.TimeDistributed(mobilenet)(input_layer)

    lstm = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.2, name='bilstm_1'))(features)
    lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=False, dropout=0.2, name='bilstm_2'))(lstm)

    x = layers.Dense(1024,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_1024'
    )(lstm2)

    x = layers.Dense(512,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_512'
    )(lstm2)

    last_embedding_layer = layers.Dense(256,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_256'
    )(x)

    # OUTPUT: Mean Rating - Single value regression
    # 
    if rating_head:
        rating_output = layers.Dense(64, activation=intermediate_activation)(last_embedding_layer)
        rating_output = layers.Dense(32, activation=intermediate_activation)(rating_output)
        rating_output = layers.Dense(1, activation='relu', name='rating')(rating_output)

        keras_loss_functions['rating'] = keras.losses.MeanSquaredError()
        keras_metrics['rating'] = keras.metrics.MeanSquaredError()
        keras_loss_weights['rating'] = 1
        network_heads.append(rating_output)

    # OUTPUT: Genres - Can be multiple, so softmax does not make sense
    # Treat each output as an individual distribution, because of that
    # use binary crossentropy
    # 
    if genre_head:
        genre_output = layers.Dense(128, activation=intermediate_activation)(last_embedding_layer)
        genre_output = layers.Dense(64, activation=intermediate_activation)(genre_output)
        genre_output = layers.Dense(n_genres, activation='sigmoid', name='genres')(genre_output)

        keras_loss_functions['genres'] = keras.losses.BinaryCrossentropy()
        keras_metrics['genres'] = keras.metrics.CategoricalAccuracy()
        keras_loss_weights['genres'] = 1
        network_heads.append(genre_output)

    # OUTPUT: Trailer Class - Can be only one, softmax!
    # 
    if class_head:
        class_output = layers.Dense(256, activation=intermediate_activation)(last_embedding_layer)
        class_output = layers.Dense(n_classes, activation='softmax', name='class')(class_output)

        keras_loss_functions['class'] = keras.losses.CategoricalCrossentropy()
        keras_metrics['class'] = keras.metrics.CategoricalAccuracy()
        keras_loss_weights['class'] = 1
        network_heads.append(class_output)

    if self_supervised_head:
        self_supervised_output = layers.Dense(np.prod(input_shape))(last_embedding_layer)
        self_supervised_output = layers.Reshape(input_shape, name='self_supervised')(self_supervised_output)

        keras_loss_functions['self_supervised'] = keras.losses.MeanSquaredError()
        keras_metrics['self_supervised'] = keras.metrics.MeanSquaredError()
        keras_loss_weights['self_supervised'] = 1
        network_heads.append(self_supervised_output)

    model = keras.Model(inputs=input_layer, outputs=network_heads)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=keras_loss_functions,
        metrics=keras_metrics,
        loss_weights=keras_loss_weights
    )

    model.summary()

    return model

def keyframe_embedding_model(n_classes=13606, n_genres=20, input_shape=(224,224,3), learning_rate=3e-4, \
    rating_head=True, genre_head=True, class_head=True, self_supervised_head=True, \
    intermediate_activation='relu', l2_beta=0):
    
    keras_loss_functions = dict()
    keras_metrics = dict()
    keras_loss_weights = dict()
    network_heads = list()
    
    input_layer = keras.Input(shape=input_shape)

    # lehl@021-05-31: Should max or average pooling be applied to the output
    # of the MobileNet network? --> "pooling" keyword
    # 
    mobilenet_feature_extractor = MobileNetV3Small(
        input_shape=input_shape,
        weights='imagenet',
        pooling='avg',
        include_top=False
    )
    mobilenet_feature_extractor.trainable = False
    x = mobilenet_feature_extractor(input_layer)

    x = layers.Dense(1024,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_1024'
    )(x)

    x = layers.Dense(512,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_512'
    )(x)

    last_embedding_layer = layers.Dense(256,
        activation=intermediate_activation,
        kernel_regularizer=tf.keras.regularizers.l2(l2_beta),
        name='dense_embedding_256'
    )(x)
 
    # OUTPUT: Mean Rating - Single value regression
    # 
    if rating_head:
        rating_output = layers.Dense(64, activation=intermediate_activation)(last_embedding_layer)
        rating_output = layers.Dense(32, activation=intermediate_activation)(rating_output)
        rating_output = layers.Dense(1, activation='relu', name='rating')(rating_output)

        keras_loss_functions['rating'] = keras.losses.MeanSquaredError()
        keras_metrics['rating'] = keras.metrics.MeanSquaredError()
        keras_loss_weights['rating'] = 1
        network_heads.append(rating_output)

    # OUTPUT: Genres - Can be multiple, so softmax does not make sense
    # Treat each output as an individual distribution, because of that
    # use binary crossentropy
    # 
    if genre_head:
        genre_output = layers.Dense(128, activation=intermediate_activation)(last_embedding_layer)
        genre_output = layers.Dense(64, activation=intermediate_activation)(genre_output)
        genre_output = layers.Dense(n_genres, activation='sigmoid', name='genres')(genre_output)

        keras_loss_functions['genres'] = keras.losses.BinaryCrossentropy()
        keras_metrics['genres'] = keras.metrics.CategoricalAccuracy()
        keras_loss_weights['genres'] = 1
        network_heads.append(genre_output)

    # OUTPUT: Trailer Class - Can be only one, softmax!
    # 
    if class_head:
        class_output = layers.Dense(256, activation=intermediate_activation)(last_embedding_layer)
        class_output = layers.Dense(n_classes, activation='softmax', name='class')(class_output)

        keras_loss_functions['class'] = keras.losses.CategoricalCrossentropy()
        keras_metrics['class'] = keras.metrics.CategoricalAccuracy()
        keras_loss_weights['class'] = 1
        network_heads.append(class_output)

    if self_supervised_head:
        self_supervised_output = layers.Dense(np.prod(input_shape))(last_embedding_layer)
        self_supervised_output = layers.Reshape(input_shape, name='self_supervised')(self_supervised_output)

        keras_loss_functions['self_supervised'] = keras.losses.MeanSquaredError()
        keras_metrics['self_supervised'] = keras.metrics.MeanSquaredError()
        keras_loss_weights['self_supervised'] = 1
        network_heads.append(self_supervised_output)

    model = keras.Model(inputs=input_layer, outputs=network_heads)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=keras_loss_functions,
        metrics=keras_metrics,
        loss_weights=keras_loss_weights
    )

    model.summary()

    return model

if __name__ == '__main__':
    keyframe_embedding_model()
    keyframe_embedding_model__bilstm()

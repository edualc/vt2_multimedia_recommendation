import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small

def keyframe_embedding_model(n_classes=64, n_genres=20, input_shape=(224,224,3), rating_head=True, genre_head=True, class_head=True, self_supervised_head=True):
    
    keras_loss_functions = dict()
    keras_metrics = dict()
    network_heads = list()
    
    input_layer = keras.Input(shape=input_shape)

    # lehl@021-05-31: Should max or average pooling be applied to the output
    # of the MobileNet network? --> "pooling" keyword
    # 
    mobilenet_feature_extractor = MobileNetV3Small(
        weights='imagenet',
        pooling='avg',
        include_top=False
    )
    mobilenet_feature_extractor.trainable = False
    x = mobilenet_feature_extractor(input_layer)

    x = layers.Dense(1024, activation='relu', name='dense_embedding_1024')(x)
    x = layers.Dense(512, activation='relu', name='dense_embedding_512')(x)
    last_embedding_layer = layers.Dense(256, activation='relu', name='dense_embedding_256')(x)
 
    # OUTPUT: Mean Rating - Single value regression
    # 
    if rating_head:
        rating_output = layers.Dense(64, activation='relu')(last_embedding_layer)
        rating_output = layers.Dense(32, activation='relu')(rating_output)
        rating_output = layers.Dense(1, activation='relu', name='rating')(rating_output)

        keras_loss_functions['rating'] = keras.losses.MeanSquaredError()
        keras_metrics['rating'] = keras.metrics.MeanSquaredError()
        network_heads.append(rating_output)

    # OUTPUT: Genres - Can be multiple, so softmax does not make sense
    # Treat each output as an individual distribution, because of that
    # use binary crossentropy
    # 
    if genre_head:
        genre_output = layers.Dense(128, activation='relu')(last_embedding_layer)
        genre_output = layers.Dense(64, activation='relu')(genre_output)
        genre_output = layers.Dense(n_genres, activation='sigmoid', name='genres')(genre_output)

        keras_loss_functions['genres'] = keras.losses.BinaryCrossentropy()
        keras_metrics['genres'] = keras.metrics.CategoricalAccuracy()
        network_heads.append(genre_output)

    # OUTPUT: Trailer Class - Can be only one, softmax!
    # 
    if class_head:
        class_output = layers.Dense(256, activation='relu')(last_embedding_layer)
        class_output = layers.Dense(n_classes, activation='softmax', name='class')(class_output)

        keras_loss_functions['class'] = keras.losses.CategoricalCrossentropy()
        keras_metrics['class'] = keras.metrics.CategoricalAccuracy()
        network_heads.append(class_output)

    if self_supervised_head:
        pass

    model = keras.Model(inputs=input_layer, outputs=network_heads)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=keras_loss_functions,
        metrics=keras_metrics
    )

    model.summary()

    return model
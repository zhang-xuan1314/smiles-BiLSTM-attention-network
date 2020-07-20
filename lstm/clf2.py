import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import tensorflow_addons

gelu = tensorflow_addons.activations.gelu
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

activation = gelu
class RemoveMask(keras.layers.Layer):
    def __init__(self, return_masked=False, no_mask=False, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.no_mask = no_mask

    def compute_mask(self, inputs, mask=None):
        return None

def AttentionWithContext(x, mask):
    x1 = layers.Dense(128, activation=activation,
                       kernel_regularizer=keras.regularizers.l2(0.005))(x)
    x1 = layers.Dropout(0.2)(x)
    x = layers.Dense(128,kernel_regularizer=keras.regularizers.l2(0.005))(x1)
    att = layers.Dense(64, activation=activation,
                       kernel_regularizer=keras.regularizers.l2(0.005))(x1)
    att = layers.Dropout(0.25)(att)
    att = layers.Dense(1)(att)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), 2)
    att = att + mask * (-1e12)
    att = layers.Softmax(1)(att)
    context_vector = att * x
    context_vector = tf.reduce_sum(context_vector, 1)
    context_vector = layers.Flatten()(context_vector)
    return context_vector

s = tf.distribute.MirroredStrategy()
with s.scope():
    inp = keras.Input(shape=[102,])
    emb = layers.Embedding(41, 32, mask_zero=True)(inp)
    mask = tf.equal(inp, 0)
    emb = layers.Masking()(emb)
    emb = layers.LayerNormalization()(emb)
    emb = layers.Dropout(0.25)(emb)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(emb)
    x = layers.Dropout(0.25)(x)

    x = AttentionWithContext(x,mask)
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.005),
                     activation=activation,
                     )(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.005),
                     activation=activation,
                     )(x)
    x = layers.Dropout(0.25)(x)
    y = layers.Dense(1, activation="sigmoid"
                     )(x)
    model = keras.Model(inputs=inp, outputs=y)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        0.005,
        10000,
        0.0001,
        power=0.5)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])

    data = np.load('aug_data.npy', allow_pickle=True)
    x_train = data[0, 0].astype('int32')
    y_train = data[0, 1].astype('int32').reshape(-1, 1)

    x_test = data[1, 0].astype('int32')

    y_test = data[1, 1].astype('int32').reshape(-1, 1)
    model.summary()
    model.fit(x_train, y_train, epochs=20, batch_size=512,
              validation_data=(x_test, y_test))


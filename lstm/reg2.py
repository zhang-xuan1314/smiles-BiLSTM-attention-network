import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
import pandas as pd
import numpy as np
import tensorflow_addons
from dataset import Dataset

gelu = tensorflow_addons.activations.gelu
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

activation = 'relu'
dropout_rate = 0.25

def myfunc(epoch,logs):
    global x_test,y_test,model,ch,test_aug_times
    y_pred = model.predict(x_test)
    y_test50 = y_test.reshape(test_aug_times, -1).mean(0)
    y_pred50 = y_pred.reshape(test_aug_times, -1).mean(0)
    r50 = r2_keras(y_test50,y_pred50)
    print(r50)
    print('\n')
    ch.append(r50.numpy())

    return r50
def r2_keras(y_true, y_pred):
    y_true = tf.reshape(y_true,(-1,1))
    y_pred = tf.reshape(y_pred,(-1,1))
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + 10e-6))

class RemoveMask(keras.layers.Layer):
    def __init__(self, return_masked=False, no_mask=False, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.no_mask = no_mask

    def compute_mask(self, inputs, mask=None):
        return None
def AttentionWithContext(x, mask):
    att = layers.Dense(64, activation=activation,
                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    att = layers.Dropout(0.25)(att)
    att = layers.Dense(1)(att)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), 2)
    att = att + mask * (-1e8)
    att = layers.Softmax(1)(att)
    context_vector = att * x
    context_vector = tf.reduce_sum(context_vector, 1)
    context_vector = layers.Flatten()(context_vector)
    return context_vector

def build_model():
    inp = keras.Input(shape=[102,],dtype=tf.int32)
    emb = layers.Embedding(41, 64, mask_zero=True,
                           embeddings_regularizer=keras.regularizers.l2(1e-5),
                           embeddings_constraint=keras.constraints.max_norm(3)
                           )(inp)
    mask = tf.equal(inp, 0)
    emb = layers.Masking(mask_value=0.0)(emb)
    emb = layers.Dropout(dropout_rate)(emb)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(emb)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = RemoveMask()(x)
    x = AttentionWithContext(x, mask)

    x = layers.Dense(256,kernel_regularizer=keras.regularizers.l2(1e-3),
                     activation=activation,
                     )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64,kernel_regularizer=keras.regularizers.l2(1e-3),
                     activation=activation
                     )(x)
    x = layers.Dropout(dropout_rate)(x)
    y = layers.Dense(1,kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    model = keras.Model(inputs=inp, outputs=y)
    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mse', optimizer=optimizer, metrics=[r2_keras])
    return  model

if __name__ == "__main__":
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['H_HT', 'Pgp_inh', 'Pgp_sub']
    keras.backend.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    tasks = ['logD']
    seeds = [7,17,23,37,43]
    for task in tasks:
        for seed in seeds:
            dataset = Dataset('data/reg/{}.txt'.format(task),'SMILES','Label',100,100,1,seed)
            test_aug_times = dataset.test_augment_times
            train_aug_times = dataset.train_augment_times
            data = dataset.get_data()
            x_train = data[0].astype('int32')
            y_train = data[1].astype('float32').reshape(-1, 1)
            y_mean = y_train.mean()
            y_max = y_train.max()
            y_train = (y_train-y_mean)/y_max

            x_test = data[2].astype('int32')
            y_test = data[3].astype('float32').reshape(-1, 1)
            y_test = (y_test-y_mean)/y_max
            cbk = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch,logs : myfunc(epoch,logs))
            model = build_model()
            model.summary()
            ch = []
            h = model.fit(x_train, y_train, epochs=150, batch_size=1024*4,
                      validation_data=(x_test, y_test),callbacks=[cbk,])
            history = h.history
            history['r3'] = ch


            res = pd.DataFrame(history)
            res.to_csv('result/{}_{}_{}_{}.csv'.format(task,seed,train_aug_times,test_aug_times))
            keras.backend.clear_session()

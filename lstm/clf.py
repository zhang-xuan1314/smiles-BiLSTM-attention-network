import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import tensorflow_addons
from dataset import Dataset
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def myfunc(epoch,logs):
    global x_test,y_test,model,ch

    y_pred = model.predict(x_test)
    y_pred = tf.math.log((y_pred+1e-8)/(1-y_pred+1e-8)).numpy()
    y_pred1 = y_pred.reshape(1,-1,1).mean(axis=0)
    y_pred1 = tf.math.sigmoid(y_pred1)
    y_test1 = y_test.reshape(1,-1,1).mean(axis=0)
    AUC = keras.metrics.AUC()
    AUC.update_state(y_test1,y_pred1)
    a2 = AUC.result().numpy()
    AUC.reset_states()

    ch.append(a2)
    print('    aug_auc:',a2)
    return a2

gelu = tensorflow_addons.activations.gelu
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

activation = gelu
dropout_rate = 0.5


class RemoveMask(keras.layers.Layer):
    def __init__(self, return_masked=False, no_mask=False, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.no_mask = no_mask

    def compute_mask(self, inputs, mask=None):
        return None

def AttentionWithContext(x, mask):
    att = layers.Dense(64, activation=activation,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l2(1e-4),
                       kernel_constraint=max_norm(2),
                       bias_constraint=max_norm(2),
                       )(x)
    att = layers.Dropout(0.25)(att)
    att = layers.Dense(1)(att)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), 2)
    att = att + mask * (-1e8)
    att = layers.Softmax(1)(att)
    context_vector = att * x
    context_vector = tf.reduce_sum(context_vector, 1)
    context_vector = layers.Flatten()(context_vector)
    return context_vector

def build_model(pn):
    inp = keras.Input(shape=(102), dtype=tf.int32)
    emb = layers.Embedding(41, 64, mask_zero=True,
                           embeddings_regularizer=keras.regularizers.l2(1e-5),
                           embeddings_constraint=keras.constraints.max_norm(3))(inp)
    mask = tf.equal(inp, tf.constant(0, dtype='int32'))
    emb = layers.Masking(mask_value=0.0)(emb)
    emb = layers.LayerNormalization(-1, beta_regularizer=keras.regularizers.l2(1e-5),
                                    gamma_regularizer=keras.regularizers.l2(1e-5)
                                    )(emb)
    emb = layers.Dropout(0.5)(emb)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                         ))(emb)
    x = layers.LayerNormalization(-1, beta_regularizer=keras.regularizers.l2(1e-5),
                                  gamma_regularizer=keras.regularizers.l2(1e-5)
                                  )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,

                                         ))(x)
    x = layers.LayerNormalization(-1, beta_regularizer=keras.regularizers.l2(1e-5),
                                  gamma_regularizer=keras.regularizers.l2(1e-5))(x)

    x = RemoveMask()(x)
    x = AttentionWithContext(x, mask)
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-3),
                     bias_regularizer=keras.regularizers.l2(1e-3),
                     activation=activation,
                     )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-3),
                     bias_regularizer=keras.regularizers.l2(1e-3),
                     activation=activation,
                     )(x)
    x = layers.Dropout(0.5)(x)
    y = layers.Dense(1, activation='sigmoid',
                     kernel_regularizer=keras.regularizers.l2(1e-3),
                     bias_regularizer=keras.regularizers.l2(1e-3),
                     )(x)
    model = keras.Model(inputs=inp, outputs=y)
    optimizer = keras.optimizers.Adam(learning_rate=0.005)

    def crosser(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
        loss = - tf.reduce_mean(y_true * tf.math.log(y_pred) + pn * (1 - y_true) * tf.math.log(1 - y_pred))
        return loss
    model.compile(loss=crosser,
                  optimizer=optimizer, metrics=['AUC'])
    return model

if __name__ == "__main__":
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['H_HT', 'Pgp_inh', 'Pgp_sub']
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    tasks = ['Ames']
    seeds = [7,17,23] #7,17,23,
    for task in tasks:
        for seed in seeds:
            dataset = Dataset('data/clf/{}.txt'.format(task),'SMILES','Label',100,20,1,seed)
            data = dataset.get_data()
            x_train = data[0].astype('int32')
            y_train = data[1].astype('int32').reshape(-1, 1)

            x_test = data[2].astype('int32')
            y_test = data[3].astype('int32').reshape(-1, 1)
            cbk = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: myfunc(epoch, logs))
            ch = []
            pn = np.sum(y_train)/(len(y_train)-np.sum(y_train))
            print(pn)

            model = build_model(pn)
            model.summary()

            abk = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,logs :myfunc(epoch,logs))

            h = model.fit(x_train, y_train, epochs=20, batch_size=512,
                      validation_data=(x_test, y_test),callbacks=[abk])

            model.summary()
            history = h.history
            history['auc'] = ch
            res = pd.DataFrame(history)
            res.to_csv('result/{}_{}.csv'.format(task,seed))
            keras.backend.clear_session()

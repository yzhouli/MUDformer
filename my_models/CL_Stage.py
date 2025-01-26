from tensorflow.keras import layers, initializers
import tensorflow as tf


class CL_Stage(layers.Layer):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self,
                 drop_rate=0.5,
                 name=None):
        super(CL_Stage, self).__init__(name=name)
        self.fc1 = tf.keras.layers.Dense(256)
        self.fc2 = tf.keras.layers.Dense(2)
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None):
        CLS_feature = inputs[:, 0, :]
        CLS_feature = self.fc1(CLS_feature)
        CLS_feature = self.dropout(CLS_feature)
        CLS_feature = self.fc2(CLS_feature)
        return CLS_feature
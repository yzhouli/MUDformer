from tensorflow.keras import layers, initializers


class MLP(layers.Layer):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self,
                 in_features,
                 mlp_ratio=4.0,
                 out_features=None,
                 drop=0.,
                 name=None):
        super(MLP, self).__init__(name=name)
        if out_features is None:
            out_features = in_features
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(out_features, name="Dense_1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x
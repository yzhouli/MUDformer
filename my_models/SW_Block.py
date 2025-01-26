from keras import layers

from my_models.SW_MHA import SW_MHA
from my_models.MLP import MLP


class SW_Block(layers.Layer):
    def __init__(self,
                 dim,
                 window_size,
                 window_stride,
                 PE='APE',
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.5,
                 name=None):
        super(SW_Block, self).__init__(name=name)
        self.drop_ratio =drop_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.mha = SW_MHA(dim,
                          PE=PE,
                          window_size=window_size,
                          window_stride=window_stride,
                          num_heads=num_heads,
                          qkv_bias=qkv_bias,
                          name='Block_MHA')
        self.block_drop = layers.Dropout(drop_ratio)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = None

    def call(self, inputs, training=None):
        inputs, mask_mat = inputs
        _, _, embed_in = inputs.shape

        inputs = self.norm1(inputs)
        inputs, mask_mat = self.mha((inputs, mask_mat), training=training)

        if not self.mlp:
            _, _, embed_mha = inputs.shape
            self.mlp = MLP(in_features=embed_mha,
                           out_features=embed_in * 2,
                           mlp_ratio=4.0,
                           drop=self.drop_ratio,
                           name="MlpBlock")
        inputs = self.block_drop(self.mlp(self.norm2(inputs)), training=training)
        return inputs, mask_mat
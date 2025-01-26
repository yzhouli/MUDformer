from tensorflow.keras import layers

from my_models.MHA import MHA
from my_models.MLP import MLP


class MHA_Block(layers.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 PE='APE',
                 qkv_bias=False,
                 drop_ratio=0.5,
                 name=None):
        super(MHA_Block, self).__init__(name=name)
        self.drop_ratio =drop_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.mha = MHA(dim, PE=PE, num_heads=num_heads, qkv_bias=qkv_bias, name='Block_MHA')
        self.block_drop = layers.Dropout(drop_ratio)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp = None

    def call(self, inputs, training=None):
        inputs, mask_mat = inputs
        inputs = self.norm1(inputs)
        inputs = self.mha((inputs, mask_mat), training=training)
        if not self.mlp:
            _, _, embedding = inputs.shape
            self.mlp = MLP(embedding, drop=self.drop_ratio, name="MlpBlock")
        inputs = inputs + self.block_drop(self.mlp(self.norm2(inputs)), training=training)
        return inputs
from keras import layers

from my_models.MHA_Block import MHA_Block
from my_models.SW_Block import SW_Block


class SW_Stage(layers.Layer):

    def __init__(self,
                 dim,
                 block_size,
                 window_size,
                 window_stride,
                 PE_li=['APE', 'APE'],
                 num_heads=8,
                 drop_ratio=0.2,
                 qkv_bias=False,
                 name=None):
        super(SW_Stage, self).__init__(name=name)
        self.num_heads = num_heads
        self.drop_ratio = drop_ratio
        self.qkv_bias = qkv_bias
        self.sw_block = SW_Block(dim=dim,
                                 PE=PE_li[0],
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 window_stride=window_stride,
                                 drop_ratio=drop_ratio,
                                 qkv_bias=qkv_bias,
                                 name='Block_SW_MHA')
        self.block_li = [MHA_Block(dim * 2,
                                   PE=PE_li[1],
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   drop_ratio=drop_ratio)
                         for i in range(block_size - 1)]


    def call(self, inputs, training=None):
        input_feature, mask_mat = inputs

        input_feature, mask_mat = self.sw_block((input_feature, mask_mat))

        for block in self.block_li:
            input_feature_new = block((input_feature, mask_mat))
            input_feature = input_feature + input_feature_new

        return input_feature, mask_mat
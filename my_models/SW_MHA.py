import numpy as np
import tensorflow as tf
from keras import layers, initializers


class SW_MHA(layers.Layer):

    def __init__(self,
                 dim,
                 window_size,
                 window_stride,
                 PE='APE',
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 n_dim = 2,
                 name=None):
        super(SW_MHA, self).__init__(name=name)
        self.PE = PE
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_dim = n_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qkv = None
        self.position_embedding = None

    def get_positional_encoding(self, inputs):
        B, seq_len, embedding_dim = inputs.shape
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        # 生成词向量维度索引 [0, 1, 2, ..., embedding_dim-1]
        dimensions = np.arange(embedding_dim)  # (embedding_dim, )
        # 计算位置编码
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(embedding_dim))
        angle_rads = positions * angle_rates  # (seq_len, embedding_dim)
        # 偶数维度使用sin，奇数维度使用cos
        pos_encoding = np.zeros((seq_len, embedding_dim))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 偶数维度
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 奇数维度

        return inputs + tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

    def create_padding_mask(self, inputs):
        # padding_positions 是一个大小为(batch_size, seq_len)的布尔矩阵
        padding_positions = tf.equal(inputs, 0)
        # 填充位置为True，非填充位置为False
        mask = tf.cast(padding_positions, tf.float32)
        return mask  # 返回大小为(batch_size, seq_len)的Mask矩阵

    def matrix_spilt(self, mat, padding_num=0):
        bitch_size, _, seq_size, embedding = mat.shape
        padding = tf.zeros(shape=(bitch_size, self.num_heads, self.window_size, embedding), dtype=tf.float32)
        padding += padding_num
        mat = tf.concat([mat, padding], axis=-2)
        mat = [mat[:, :, i:i + self.window_size, :]
               for i in range(seq_size -1, -1, -self.window_stride)]
        mat = tf.stack(mat)
        re_shape = [i + 1 for i in range(len(mat.shape)-1)]
        re_shape.insert(self.n_dim, 0)
        mat = tf.transpose(mat, re_shape)
        return mat

    def build_mask(self, mask_mat):
        mask_mat = [mask_mat for i in range(self.num_heads)]
        mask_mat = tf.transpose(mask_mat, [1, 0, 2])
        mask_mat = tf.expand_dims(mask_mat, -1)
        mask_mat = self.matrix_spilt(mask_mat, padding_num=-100)
        mask = tf.matmul(a=mask_mat, b=mask_mat, transpose_b=True)
        mask_mat = mask_mat[:, 0, :, :, 0]
        mask_mat = tf.reduce_mean(mask_mat, axis=-1, keepdims=False)
        return mask, mask_mat

    def call(self, inputs, training=None):
        inputs, mask_mat = inputs
        mask, mask_mat = self.build_mask(mask_mat=mask_mat)

        B, N, C = inputs.shape
        if self.PE == 'APE':
            inputs = self.get_positional_encoding(inputs)
        elif self.PE == 'TPE':
            if self.position_embedding is None:
                self.position_embedding = tf.Variable(
                    initial_value=tf.random.normal((N, C)),  # 使用随机数初始化
                    trainable=True  # 设置为可训练
                )
            inputs += self.position_embedding

        if not self.qkv:
            self.qkv = layers.Dense(C * 3, use_bias=self.qkv_bias, name="qkv",
                                kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.matrix_spilt(q)
        k = self.matrix_spilt(k)
        v = self.matrix_spilt(v)

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = attn - mask
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)

        x = tf.transpose(x, [0, 2, 1, 3, 4])
        x = x[:, :, :, 0, :]
        B, L, H, E = x.shape
        x = tf.reshape(x, [B, L, H * E])

        return x, mask_mat

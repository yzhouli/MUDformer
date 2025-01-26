import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers


class MHA(layers.Layer):

    def __init__(self,
                 dim,
                 num_heads=8,
                 PE='APE',
                 qkv_bias=False,
                 qk_scale=None,
                 name=None):
        super(MHA, self).__init__(name=name)
        self.num_heads = num_heads
        self.PE = PE
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qkv_1 = None
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

    def call(self, inputs, training=None):
        inputs, mask_mat = inputs
        mask_mat = tf.expand_dims(mask_mat, -1)
        mask_mat = tf.matmul(a=mask_mat, b=mask_mat, transpose_b=True)
        mask_mat = -mask_mat
        mask_mat = [mask_mat for i in range(self.num_heads)]
        mask_mat = tf.transpose(mask_mat, [1, 0, 2, 3])

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

        if not self.qkv_1:
            self.qkv_1 = layers.Dense(C * 3, use_bias=self.qkv_bias, name="qkv",
                                kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())
        qkv = self.qkv_1(inputs)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = attn + mask_mat
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)

        x = tf.transpose(x, [0, 2, 1, 3])

        x = tf.reshape(x, [B, N, C])

        return x

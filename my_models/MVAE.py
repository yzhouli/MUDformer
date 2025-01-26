import tensorflow as tf
from tensorflow.keras import layers

class MVAE(layers.Layer):
    def __init__(self, input_dim=768, latent_dim=64):
        super(MVAE, self).__init__()

        # 文本模态编码器
        self.text_encoder = layers.Dense(256, activation='relu')
        self.text_mean = layers.Dense(latent_dim)
        self.text_log_var = layers.Dense(latent_dim)

        # 图像模态编码器
        self.image_encoder = layers.Dense(256, activation='relu')
        self.image_mean = layers.Dense(latent_dim)
        self.image_log_var = layers.Dense(latent_dim)

        # 文本解码器
        self.text_decoder = layers.Dense(input_dim, activation='relu')
        self.text_decoder_output = layers.Dense(input_dim, activation='sigmoid')  # Sigmoid for reconstruction

        # 图像解码器
        self.image_decoder = layers.Dense(input_dim, activation='relu')
        self.image_decoder_output = layers.Dense(input_dim, activation='sigmoid')  # Sigmoid for reconstruction

    def encode(self, x, mod_type='text'):
        mean, log_var = None, None
        if mod_type == 'text':
            x = self.text_encoder(x)
            mean = self.text_mean(x)
            log_var = self.text_log_var(x)
        elif mod_type == 'image':
            x = self.image_encoder(x)
            mean = self.image_mean(x)
            log_var = self.image_log_var(x)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        # 重参数化技巧
        eps = tf.random.normal(shape=tf.shape(mean))
        std = tf.exp(0.5 * log_var)
        return mean + eps * std

    def decode(self, z, mod_type='text'):
        output = None
        if mod_type == 'text':
            z = self.text_decoder(z)
            output = self.text_decoder_output(z)
        elif mod_type == 'image':
            z = self.image_decoder(z)
            output = self.image_decoder_output(z)
        return output

    def call(self, inputs, training=None):

        text_input, image_input = inputs

        # 编码
        text_mean, text_log_var = self.encode(text_input, mod_type='text')
        image_mean, image_log_var = self.encode(image_input, mod_type='image')

        # 跨模态特征整合
        mean = tf.concat([text_mean, image_mean], axis=-1)
        log_var = tf.concat([text_log_var, image_log_var], axis=-1)

        # 使用重参数化技巧获得潜在变量
        z = self.reparameterize(mean, log_var)

        # 解码
        reconstructed_text = self.decode(z, mod_type='text')
        reconstructed_image = self.decode(z, mod_type='image')

        return z, reconstructed_text, reconstructed_image, mean, log_var

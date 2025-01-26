import tensorflow as tf
from keras import layers

from my_models.MVAE import MVAE


class MVAE_Stage(layers.Layer):

    def __init__(self,
                 dim,
                 latent_dim,
                 name=None):
        super(MVAE_Stage, self).__init__(name=name)
        self.mvae = MVAE(input_dim=dim, latent_dim=latent_dim)

    def call(self, inputs, training=None):
        image_input, text_input = inputs

        bitch, seq_len, embedding_dim = image_input.shape
        image_input = tf.reshape(image_input, [bitch * seq_len, embedding_dim])
        text_input = tf.reshape(text_input, [bitch * seq_len, embedding_dim])

        input_feature, reconstructed_text, reconstructed_image, mean, log_var = self.mvae((text_input, image_input))
        _, embedding_dim = input_feature.shape
        input_feature = tf.reshape(input_feature, [bitch, seq_len, embedding_dim])
        _, embedding_dim = reconstructed_image.shape
        reconstructed_image = tf.reshape(reconstructed_image, [bitch, seq_len, embedding_dim])
        _, embedding_dim = reconstructed_text.shape
        reconstructed_text = tf.reshape(reconstructed_text, [bitch, seq_len, embedding_dim])
        _, embedding_dim = mean.shape
        mean = tf.reshape(mean, [bitch, seq_len, embedding_dim])
        _, embedding_dim = log_var.shape
        log_var = tf.reshape(log_var, [bitch, seq_len, embedding_dim])
        MVAE_data = reconstructed_text, reconstructed_image, mean, log_var

        return input_feature, MVAE_data

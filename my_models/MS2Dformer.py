import tensorflow as tf
from tensorflow.keras import Model

from my_models.CL_Stage import CL_Stage
from my_models.MVAE_Stage import MVAE_Stage
from my_models.SW_Stage import SW_Stage


class MS2Dformer(Model):

    def __init__(self,
                 dim,
                 latent_dim,
                 window_size_li,
                 window_stride_li,
                 block_size_li,
                 PE_li=['APE', 'APE'],
                 num_heads=8,
                 drop_ratio=0.2,
                 qkv_bias=False,
                 name=None):
        super(MS2Dformer, self).__init__(name=name)
        self.stage_1 = MVAE_Stage(dim=dim, latent_dim=latent_dim, name='Stage_1')

        self.stage_2 = SW_Stage(dim=latent_dim * 2,
                                num_heads=num_heads,
                                PE_li=PE_li,
                                window_size=window_size_li[0],
                                window_stride=window_stride_li[0],
                                block_size=block_size_li[0],
                                drop_ratio=drop_ratio,
                                qkv_bias=qkv_bias,
                                name='Stage_2')

        self.stage_3 = SW_Stage(dim=latent_dim * 4,
                                num_heads=num_heads,
                                PE_li=PE_li,
                                window_size=window_size_li[1],
                                window_stride=window_stride_li[1],
                                block_size=block_size_li[1],
                                drop_ratio=drop_ratio,
                                qkv_bias=qkv_bias,
                                name='Stage_2')

        self.stage_4 = CL_Stage(drop_rate=drop_ratio, name='Stage_4')

    def call(self, inputs, training=None):
        image_mat, text_mat, mask_mat = inputs

        input_feature, MVAE_data = self.stage_1((image_mat, text_mat))

        input_feature, mask_mat = self.stage_2((input_feature, mask_mat))

        input_feature, mask_mat = self.stage_3((input_feature, mask_mat))

        CLS_feature = self.stage_4(input_feature)

        return CLS_feature, MVAE_data

    def compute_loss(self, class_out, class_label, image_input, text_input, MVAE_data):
        reconstructed_text, reconstructed_image, mean, log_var = MVAE_data

        class_loss = tf.losses.categorical_crossentropy(class_label, class_out, from_logits=True)
        class_loss = tf.reduce_mean(class_loss)

        # 重建损失
        reconstruction_loss_text = tf.reduce_mean(tf.reduce_sum(tf.square(reconstructed_text - text_input), axis=-1))
        reconstruction_loss_image = tf.reduce_mean(tf.reduce_sum(tf.square(reconstructed_image - image_input), axis=-1))

        # KL散度
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1))

        MVAE_loss = reconstruction_loss_text + reconstruction_loss_image + kl_loss

        return MVAE_loss, class_loss
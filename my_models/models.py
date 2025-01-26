from my_models.MS2Dformer import MS2Dformer


class Models(object):

    MS2Dformer_Base = MS2Dformer(dim=768,
                                 latent_dim=16,
                                 PE_li=['APE', 'APE'],
                                 window_size_li=[64, 64],
                                 window_stride_li=[1, 1],
                                 block_size_li=[3, 3],
                                 num_heads=8,
                                 drop_ratio=0.2,
                                 qkv_bias=True,
                                 name='MS2Dformer_Base')

    MS2Dformer_Middle = MS2Dformer(dim=768,
                                 latent_dim=16,
                                   PE_li=['APE', 'APE'],
                                 window_size_li=[128, 64],
                                 window_stride_li=[32, 4],
                                 block_size_li=[3, 11],
                                 num_heads=8,
                                 drop_ratio=0.2,
                                 qkv_bias=True,
                                 name='MS2Dformer_Middle')

    MS2Dformer_Large = MS2Dformer(dim=768,
                                  latent_dim=64,
                                  PE_li=['APE', 'APE'],
                                  window_size_li=[128, 64],
                                  window_stride_li=[32, 4],
                                  block_size_li=[7, 17],
                                  num_heads=8,
                                  drop_ratio=0.2,
                                  qkv_bias=True,
                                  name='MS2Dformer_Large')
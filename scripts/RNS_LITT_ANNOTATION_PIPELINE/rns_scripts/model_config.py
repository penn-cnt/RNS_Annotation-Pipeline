class MODEL_CONFIG:
    """
    This class contains all the configuration parameters for the project.
    """
    spatial_transformer_blocks = 6  # number of transformer blocks
    spatial_transformer_hidden = 256  # transformer hidden size
    spatial_transformer_heads = 6  # transformer heads
    spatial_transformer_inner_heads = 64  # transformer inner size
    spatial_transformer_mlp_dim = 1024

    # default wav2vec2 config
    # transformer (implement base transformer)
    temporal_transformer_blocks = 12  # number of transformer blocks
    temporal_transformer_hidden = 256  # transformer hidden size
    # transformer_hidden    = 768 // 16   # transformer hidden size
    temporal_transformer_heads = 8  # transformer heads
    temporal_transformer_inner_heads = 64  # transformer inner size
    temporal_transformer_mlp_dim = 1024
    dropout = 0.1  # dropout
    activation = "gelu"  # activation function

    # feature extractor
    ft_enc_dims = (16, 64, 128, 256)
    ft_enc_strides = (2, 2, 2, 1)
    ft_enc_kernel_widths = (3, 3, 2, 2)

    channel_buffer_size = 4  # number of channels to buffer

    # training parameters
    batch_size = 16  # batch size # larger batch size helps
    learning_rate = 1e-3  # learning rate
    epochs = 500  # number of epochs


    x_max = 300
    y_max = 300
    z_max = 300

    prototype_dim = 128
    prototype_n = 512

    multi_crop_config = [6000, 6000, 2000, 2000, 2000, 2000]

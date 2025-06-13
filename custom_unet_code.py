import tensorflow as tf 
from tensorflow.keras import layers, models

def time_distributed_conv_block(input_tensor, num_filters):
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(input_tensor)
    x = layers.ReLU()(x)
    
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = layers.ReLU()(x)
    return x

def time_distributed_encoder_block_resp(input_tensor, num_filters, temporal_maxpool=True):
    x = time_distributed_conv_block(input_tensor, num_filters)

    p = layers.MaxPooling3D((1, 4, 4))(x)

    if temporal_maxpool:

        p = tf.transpose(p, (0,2,3,1,4))
        p2 = layers.TimeDistributed(layers.TimeDistributed(layers.MaxPooling1D((2))))(p)
        p2 = tf.transpose(p2, (0,3,1,2,4))
        return x, p2
    else:
        return x, p

def time_distributed_decoder_block_resp(input_tensor, skip_tensor, num_filters):
    
    x = layers.TimeDistributed(layers.UpSampling2D(( 4, 4), interpolation='bilinear'))(input_tensor)
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(x)

    x = layers.Concatenate()([x, skip_tensor])
    x = time_distributed_conv_block(x, num_filters)
    return x

def build_3d_unet_resp(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoding path
    s1, p1 = time_distributed_encoder_block_resp(inputs, 32,temporal_maxpool=False)
    s2, p2 = time_distributed_encoder_block_resp(p1, 64,temporal_maxpool=False)

    # Bridge
    b1 = time_distributed_conv_block(p2, 128)

    d1 = time_distributed_decoder_block_resp(b1, s2, 64)
    d2 = time_distributed_decoder_block_resp(d1, s1, 32)

    outputs = layers.Conv3D(num_classes, (1, 1, 1))(d2)

    model = models.Model(inputs, outputs, name="3D-U-Net-resp")
    return model

def time_distributed_encoder_block(input_tensor, num_filters, temporal_maxpool=True):
    x = time_distributed_conv_block(input_tensor, num_filters)

    p = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    if temporal_maxpool:

        p = tf.transpose(p, (0,2,3,1,4))
        p2 = layers.TimeDistributed(layers.TimeDistributed(layers.MaxPooling1D((2))))(p)
        p2 = tf.transpose(p2, (0,3,1,2,4))
        return x, p2
    else:
        return x, p

def time_distributed_decoder_block(input_tensor, skip_tensor, num_filters, temporal_upsamp=True):
    x = layers.TimeDistributed(layers.UpSampling2D(( 2, 2)))(input_tensor)
    x = layers.TimeDistributed(layers.Conv2D(num_filters, (3, 3), padding="same"))(x)
    if temporal_upsamp:
        x = tf.transpose(x, (0,2,3,1,4))
        x = layers.TimeDistributed(layers.TimeDistributed(layers.UpSampling1D((2))))(x)
        x = layers.TimeDistributed(layers.TimeDistributed(layers.Conv1D(num_filters, (2),padding="same")))(x)
        x = tf.transpose(x, (0,3,1,2,4))

    if x.shape[4] == 64:
        skip_tensor = tf.transpose(skip_tensor, (0,2,3,1,4))
        skip_tensor = layers.TimeDistributed(layers.TimeDistributed(layers.Conv1DTranspose(num_filters,kernel_size=2,strides=2)))(skip_tensor)
        skip_tensor = tf.transpose(skip_tensor, (0,3,1,2,4))

    if x.shape[4] == 32:
        skip_tensor = tf.transpose(skip_tensor, (0,2,3,1,4))
        skip_tensor = layers.TimeDistributed(layers.TimeDistributed(layers.Conv1DTranspose(num_filters,kernel_size=2,strides=2)))(skip_tensor)
        skip_tensor = layers.TimeDistributed(layers.TimeDistributed(layers.Conv1DTranspose(num_filters,kernel_size=2,strides=2)))(skip_tensor)
        skip_tensor = tf.transpose(skip_tensor, (0,3,1,2,4))

    x = layers.Concatenate()([x, skip_tensor])
    x = time_distributed_conv_block(x, num_filters)
    return x

def build_3d_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoding path
    s1, p1 = time_distributed_encoder_block(inputs, 32,temporal_maxpool=False)
    s2, p2 = time_distributed_encoder_block(p1, 64,temporal_maxpool=False)
    s3, p3 = time_distributed_encoder_block(p2, 128,temporal_maxpool=True)

    # Bridge
    b1 = time_distributed_conv_block(p3, 256)

    # Decoding path
    d1 = time_distributed_decoder_block(b1, s3, 128,temporal_upsamp=True)
    d2 = time_distributed_decoder_block(d1, s2, 64,temporal_upsamp=True)
    d3 = time_distributed_decoder_block(d2, s1, 32,temporal_upsamp=True)

    # Output layer
    outputs = layers.Conv3D(num_classes, (1, 1, 1))(d3)

    model = models.Model(inputs, outputs, name="3D-U-Net")
    return model
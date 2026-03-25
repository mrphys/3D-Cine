import tensorflow as tf 
from tensorflow.keras import layers, models

def conv_block(input_tensor, num_filters):
		x = layers.Conv3D(num_filters, (3, 3, 3), padding="same", kernel_initializer="he_normal")(input_tensor)
		x = layers.LayerNormalization()(x) 
		x = layers.LeakyReLU(0.1)(x)
		x = layers.Conv3D(num_filters, (3, 3, 3), padding="same", kernel_initializer="he_normal")(x)
		x = layers.LayerNormalization()(x) 
		x = layers.LeakyReLU(0.1)(x)
		return x

def encoder_block_resp(input_tensor, num_filters):
	x = conv_block(input_tensor, num_filters)
	p = layers.MaxPooling3D((1, 2, 2))(x)
 
	return x, p

def decoder_block_resp(input_tensor, skip_tensor, num_filters):
	x = layers.Conv3DTranspose(num_filters, (1,4,4), strides=(1,2,2), padding="same",kernel_initializer="he_normal")(input_tensor)
	x = layers.LayerNormalization()(x)
	x = layers.LeakyReLU(0.1)(x)

	x = layers.Concatenate()([x, skip_tensor])
	x = conv_block(x, num_filters)
	return x

def build_3d_unet_resp(input_shape, num_classes):
	inputs = layers.Input(shape=input_shape)

	# Encoding path
	s1, p1 = encoder_block_resp(inputs, 32)
	s2, p2 = encoder_block_resp(p1, 64)
	s3, p3 = encoder_block_resp(p2, 128)

	# Bridge
	b1 = conv_block(p3, 256)
 
	d1 = decoder_block_resp(b1, s3, 128)
	d2 = decoder_block_resp(d1, s2, 64)
	d3 = decoder_block_resp(d2, s1, 32)

	outputs = layers.Conv3D(num_classes, (1, 1, 1))(d3)

	model = models.Model(inputs, outputs, name="3D-U-Net-resp")
	return model

def encoder_block_sr(input_tensor, num_filters, temporal_maxpool=True):
	x = conv_block(input_tensor, num_filters)
	if not temporal_maxpool:
		p = layers.MaxPooling3D((1, 2, 2))(x)
	if temporal_maxpool:
		p = layers.MaxPooling3D((2, 2, 2))(x)

	return x, p

def decoder_block_SR(input_tensor, skip_tensor, num_filters):
	x = layers.Conv3DTranspose(num_filters, (4,4,4), strides=(2,2,2), padding="same",kernel_initializer="he_normal")(input_tensor)
	x = layers.LayerNormalization()(x)
	x = layers.LeakyReLU(0.1)(x)
	
	if skip_tensor.shape[4] == 64:
		skip_tensor = layers.UpSampling3D((2,1,1))(skip_tensor)

	if skip_tensor.shape[4] == 32:
		skip_tensor = layers.UpSampling3D((4,1,1))(skip_tensor)
		
	x = layers.Concatenate()([x, skip_tensor])
	x = conv_block(x, num_filters)
	return x


def build_3d_unet(input_shape, num_classes):
	inputs = layers.Input(shape=input_shape)
	# Encoding path
	s1, p1 = encoder_block_sr(inputs, 32, temporal_maxpool=False)
	s2, p2 = encoder_block_sr(p1, 64, temporal_maxpool=False)
	s3, p3 = encoder_block_sr(p2, 128, temporal_maxpool=True)
	# Bridge
	b1 = conv_block(p3, 256)
	# Decoding path
	d1 = decoder_block_SR(b1, s3, 128)
	d2 = decoder_block_SR(d1, s2, 64)
	d3 = decoder_block_SR(d2, s1, 32)
	# Output layer
	outputs = layers.Conv3D(num_classes, (1, 1, 1))(d3)
	model = models.Model(inputs, outputs, name="3D-U-Net")
	return model
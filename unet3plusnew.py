# Python file containing unet3plus code used to train segmentation model

import tensorflow as tf
import numpy as np
import layer_util
tf.random.set_seed(0)

class unet3plus:
   def __init__(self,
                inputs,
                filters = [32,64,128,256,512],
                rank = 2,
                out_channels = 3,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_regularizer=None,
                bias_regularizer=None,
                add_dropout = False,
                padding = 'same',
                dropout_rate = 0.5,
                kernel_size = 3,
                out_kernel_size = 3,
                pool_size = 2,
                encoder_block_depth = 2,
                decoder_block_depth = 1,
                batch_norm = True,
                activation = 'relu',
                out_activation = None,
                skip_batch_norm = True,
                skip_type = 'encoder',
                CGM = False,
                deep_supervision = True):
       
       
       self.inputs = inputs
       self.filters = filters
       self.scales = len(filters)
       self.rank = rank
       self.out_channels = out_channels
       self.encoder_block_depth = encoder_block_depth
       self.decoder_block_depth = decoder_block_depth
       self.kernel_size = kernel_size
       self.add_dropout = add_dropout
       self.dropout_rate = dropout_rate
       self.skip_type = skip_type  
       self.skip_batch_norm = skip_batch_norm
       self.batch_norm = batch_norm
       if isinstance(activation, str):
           self.activation = tf.keras.activations.get(activation)
       else:
           self.activation = activation
       if isinstance(out_activation, str):
           self.out_activation = tf.keras.activations.get(out_activation)
       else:
           self.out_activation = out_activation
       # Assign pool size
       if isinstance(pool_size,tuple):
           self.pool_size = pool_size
       else:
           self.pool_size = tuple([pool_size for _ in range(rank)])
       if isinstance(kernel_size,tuple):
           self.kernel_size = kernel_size
       else:
           self.kernel_size = tuple([kernel_size for _ in range(rank)])
       if isinstance(out_kernel_size,tuple):
           self.out_kernel_size = out_kernel_size
       else:
           self.out_kernel_size = tuple([out_kernel_size for _ in range(rank)])
       self.CGM = CGM
       self.deep_supervision = deep_supervision
       self.conv_config = dict(kernel_size = self.kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
       self.out_conv_config = dict(kernel_size = out_kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
   
   def aggregate(self, scale_list, scale):
       X = tf.keras.layers.Concatenate(name = f'D{scale}_input', axis = -1)(scale_list)
       X = self.conv_block(X, self.filters[0] * self.scales, num_stacks = self.decoder_block_depth, layer_type = 'Decoder', scale=scale)
       return X
   
   def deep_sup(self, inputs, scale):
       conv = layer_util.get_nd_layer('Conv', self.rank)
       upsamp = layer_util.get_nd_layer('UpSampling', self.rank)
       size = tuple(np.array(self.pool_size)** (abs(scale-1)))
       if self.rank == 2:
           upsamp_config = dict(size=size, interpolation='bilinear')
       else:
           upsamp_config = dict(size=size)  
       X = inputs  
       X = conv(self.out_channels, activation = None, **self.out_conv_config, name = f'deepsup_conv_{scale}')(X)
       if scale != 1:
           X = upsamp(**upsamp_config, name = f'deepsup_upsamp_{scale}')(X)
       #X = tf.keras.layers.Activation(activation = 'sigmoid' if self.out_channels == 1 else 'softmax', name = f'deepsup_activation_{scale}')(X)
       X =self.out_activation(X)
       return X
       
       
       
   def full_scale(self, inputs, to_layer, from_layer):
       conv = layer_util.get_nd_layer('Conv', self.rank)
       layer_diff = from_layer - to_layer  
       size = tuple(np.array(self.pool_size)** (abs(layer_diff)))
       maxpool = layer_util.get_nd_layer('MaxPool', self.rank)
       upsamp = layer_util.get_nd_layer('UpSampling', self.rank)
       if self.rank == 2:
           upsamp_config = dict(size=size, interpolation='bilinear')
       else:
           upsamp_config = dict(size=size)
       
       X = inputs        
       if to_layer < from_layer:
           X = upsamp(**upsamp_config, name = f'Skip_Upsample_{from_layer}_{to_layer}')(X)
       elif to_layer > from_layer:
           X = maxpool(pool_size = size, name = f'Skip_Maxpool_{from_layer}_{to_layer}')(X)
       
       if self.skip_batch_norm:
           X = self.conv_block(X, self.filters[0], num_stacks = self.decoder_block_depth, layer_type ='Skip', scale = f'{from_layer}_{to_layer}')
       else:
           X = conv(self.filters[0],**self.conv_config, name = f'Skip_Conv_{from_layer}_{to_layer}')(X)
           
       return X
       
   def conv_block(self, inputs, filters, num_stacks,layer_type, scale):
       conv = layer_util.get_nd_layer('Conv', self.rank)
       X = inputs
       for i in range(num_stacks):
           X = conv(filters, **self.conv_config, name = f'{layer_type}{scale}_Conv_{i+1}')(X)
           if self.batch_norm:
               X = tf.keras.layers.BatchNormalization(axis=-1, name = f'{layer_type}{scale}_BN_{i+1}')(X)
           #X = tf.keras.layers.LeakyReLU(name = f'{layer_type}{scale}_Activation_{i+1}')(X)
           X =  self.activation(X)
       return X
   
   
   def encode(self, inputs, scale, num_stacks):
       maxpool = layer_util.get_nd_layer('MaxPool', self.rank)
       scale -= 1 # python index
       filters = self.filters[scale]
       X = inputs
       if scale != 0:
           X = maxpool(pool_size=self.pool_size, name = f'encoding_{scale}_maxpool')(X)
       X = self.conv_block(X, filters, num_stacks, layer_type = 'Encoder', scale = scale+1)
       if scale == (self.scales-1) and self.add_dropout:
           X = tf.keras.layers.Dropout(rate = self.dropout_rate, name = f'Encoder{scale+1}_dropout')(X)
       return X
       
   def outputs(self):
       XE  = [self.inputs]
       for i in range(self.scales):
           XE.append(self.encode(XE[i], scale = i+1, num_stacks = self.encoder_block_depth))
       XD = [XE[-1]]
       if self.skip_type == 'encoder':
           for decoder_level in range(self.scales-1,0,-1):
               input_contributions = []
               for unet_level in range(1,self.scales+1):
                   if unet_level == decoder_level+1:
                       input_contributions.append(self.full_scale(XD[-1], decoder_level, unet_level))
                   else:
                       input_contributions.append(self.full_scale(XE[unet_level], decoder_level, unet_level))
               XD.append(self.aggregate(input_contributions,decoder_level))
       elif self.skip_type == 'decoder':
           for decoder_level in range(self.scales-1,0,-1):
               skip_contributions = []
               # Append skips from encoder
               for encoder_level in range(1,decoder_level+1):
                   skip_contributions.append(self.full_scale(XE[encoder_level], decoder_level, encoder_level))
               # Append skips from decoder
               for i in range(len(XD)-1,-1,-1):
                   skip_contributions.append(self.full_scale(XD[i], decoder_level, (self.scales-i)))
               XD.append(self.aggregate(skip_contributions,decoder_level))
       else:
           raise ValueError(f"Invalid skip_type")
       if self.deep_supervision == True:
           XD = [self.deep_sup(xd, self.scales-i) for i,xd in enumerate(XD)]
           return XD
       else:
           XD[-1] = self.deep_sup(XD[-1],1)
           return XD[-1]
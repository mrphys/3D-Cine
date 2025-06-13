## Python file for training respiratory correction model

# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow_mri as tfmri
import tensorflow as tf
import tensorflow_addons as tfa
import random
import gradient_losses
from process_utils import *
from custom_unet_code import *

# Adds patient numbers for MMWHS and HVSMR data
mmwhs_number = 10 # Change to reflect number of augmented MMWHS datasets
hvsmr_number = 10 # Change to reflect number of augmented HVSMR datasets
patients_list = []
for i in range(mmwhs_number+hvsmr_number):
   patients_list.append(i)

random.Random(4).shuffle(patients_list) #Set seed shuffle for consistency

class CustomDataGen():    
    def __init__(self, 
                 patients,
                 cohort
                ):
        self.patients = patients
        self.cohort = cohort
        
    def data_generator(self):

        for patient in self.patients:

            #Load MMWHS data
            if patient >= hvsmr_number:
              image = np.load(f"./training_data/Low_Res_resp_MMWHS_{patient-hvsmr_number}.npy").astype('float32') # Data with respiratory motion
              target = np.load(f"./training_data/Low_Res_MMWHS_{patient-hvsmr_number}.npy").astype('float32') # Low res data, no repiratory motion 

            #Load HVSMR data
            else:
              image = np.load(f"./training_data/Low_Res_resp_HVSMR_{patient}.npy").astype('float32') #  Data with respiratory motion
              target = np.load(f"./training_data/Low_Res_HVSMR_{patient}.npy").astype('float32') # Low res data, no repiratory motion 

            yield norm(image), norm(target)  # Normalise both datasets before outputs

    def get_gen(self):
        return self.data_generator()
    
output_types = (tf.float32,tf.float32) # Specify your input and output types

patient_total =  len(patients_list) 
trai_pat_num = (int(np.rint((92/100)*patient_total))) # Select percentage of training data (rest is validation)
train_patients = patients_list[:trai_pat_num]
val_patients = patients_list[trai_pat_num:]

# Define your input and output data shape
image_size = [28,256,128,1]
image_in = [28,256,128,1]
input_shape = image_in
output_shape = image_size
train_gen = CustomDataGen(train_patients, 'train').get_gen
val_gen   = CustomDataGen(val_patients, 'val').get_gen
 
output_signature = (tf.TensorSpec(shape=input_shape, dtype=tf.float32), 
                tf.TensorSpec(shape=output_shape, dtype=tf.float32))
 
train_ds = tf.data.Dataset.from_generator(train_gen, 
                                          output_signature = output_signature)
 
val_ds = tf.data.Dataset.from_generator(val_gen, 
                                        output_signature = output_signature)
 
BATCH_SIZE = 1
train_ds = train_ds.shuffle(patient_total, seed = 42, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(-1)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(-1)

# Define gradfient mean absolute error loss
class MAEGrad(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="MAEGrad")
  def mean_absolute_error(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)
  def call(self, y_true, y_pred):

    grad_true = tfmri.image.image_gradients(
    y_true, method='sobel', norm=norm,
    batch_dims = None, image_dims=3)

    grad_pred = tfmri.image.image_gradients(
    y_pred, method='sobel', norm=norm,
    batch_dims=None, image_dims=3)

    mae = self.mean_absolute_error(grad_true, grad_pred)

    return mae

# Define mean absolute error loss
class CustomMAE(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="MAEResp")
  def call(self, y_true, y_pred):
    
    mae = tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)

    return mae

# Define regularization loss for predicted deformation fields
class RegLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="RegLoss")
  def call(self, y_true, y_pred):

    deformation_field = y_pred
    
    # Reshape to combine batch and slices dimensions for gradient computation
    shape = tf.shape(deformation_field)
    B, S, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    
    # Reshape to (B * S, H, W, C) to apply tf.image.sobel_edges to each 2D slice independently
    deformation_field_reshaped = tf.reshape(deformation_field, [B * S, H, W, C])
    
    # Compute gradients along height and width for each displacement component
    grad_x = tf.image.sobel_edges(deformation_field_reshaped[..., 0:1])  # Gradient for x-component
    grad_y = tf.image.sobel_edges(deformation_field_reshaped[..., 1:2])  # Gradient for y-component
    
    # Extract gradient components for each displacement component
    grad_x_x, grad_x_y = grad_x[..., 0], grad_x[..., 1]
    grad_y_x, grad_y_y = grad_y[..., 0], grad_y[..., 1]
    
    # Calculate squared L2 norm for each gradient component
    grad_norm_squared = (
        tf.reduce_sum(tf.square(grad_x_x) + tf.square(grad_x_y)) +
        tf.reduce_sum(tf.square(grad_y_x) + tf.square(grad_y_y))
    )
    
    # Compute the mean loss over all slices and batch
    loss = tf.reduce_mean(grad_norm_squared) / tf.cast(B * S, tf.float32)
    
    return 5*loss*10e-8  

# Define overall loss with weights lambda_1 and lambda_2
lambda_1 = 2.6786818474503695
lambda_2 = 0.5
sum_loss = gradient_losses.WeightedSumLoss([MAEGrad(),CustomMAE()],[lambda_1,lambda_2])


# Define lambda layer to apply predicted deformations to input slices
def deformation(x):

    sagittal_deformed = []
    for i in range(28):
        
        input_img = tf.expand_dims(x[0][0,i,:,:], -1) 
        dy = tf.expand_dims(x[1][0,i,:,:], -1)
        dx = tf.expand_dims(x[2][0,i,:,:], -1)

        displacement = tf.concat((dy,dx), axis=-1)

        img = tf.image.convert_image_dtype(tf.expand_dims(input_img, 0), tf.dtypes.float32)
        displacement = tf.image.convert_image_dtype(displacement, tf.dtypes.float32)
        dense_img_warp = tfa.image.dense_image_warp(img, displacement)
        im_deformed = tf.squeeze(dense_img_warp, 0)
        sagittal_deformed.append(im_deformed)

    sagittal_deformed = tf.image.convert_image_dtype(sagittal_deformed, tf.dtypes.float32)
    sagittal_deformed = tf.expand_dims(sagittal_deformed,axis= 0)

    return sagittal_deformed

# Build and compile UNet model
tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape = [28,256,128,1])
unet = build_3d_unet_resp([28,256,128,1],num_classes =2) # Acts as a deformation field generator
deformation_fields = unet(inputs) # Outputs the deformation fields
lambda_deformation = tf.keras.layers.Lambda(deformation)
out_2 = lambda_deformation([inputs[:,:,:,:,0],deformation_fields[:,:,:,:,0],deformation_fields[:,:,:,:,1]]) # Outputs the deformed volume
outputs  = [deformation_fields,out_2] 
complete_model = tf.keras.Model(inputs = inputs, outputs = outputs)
lr_schedule = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
complete_model.compile(optimizer=optimizer, loss=[RegLoss(),sum_loss])
complete_model.summary()

num_epochs = 500
model_name = 'Respiratory_correction_model'

log_dir = "logs/fit/" + model_name
callbacks=[tf.keras.callbacks.ModelCheckpoint('./models_final/' + model_name, save_best_only=True, monitor='val_loss') ,
          tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)]

history = complete_model.fit(train_ds ,validation_data=val_ds,epochs=num_epochs, verbose=1, callbacks= callbacks)
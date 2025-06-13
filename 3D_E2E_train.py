## Python file for training end-to-end respiratory correction and super-resolution models

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
from tensorflow.keras import layers

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
              image = np.load(f"./training_data/Low_Res_resp_MMWHS_{patient-hvsmr_number}.npy").astype('float32')  # Data with respiratory motion
              target = np.load(f"./training_data/High_Res_MMWHS_{patient-hvsmr_number}.npy").astype('float32') # High res data

            #Load HVSMR data
            else:
              image = np.load(f"./training_data/Low_Res_resp_HVSMR_{patient}.npy").astype('float32') #  Data with respiratory motion
              target = np.load(f"./training_data/High_Res_HVSMR_{patient}.npy").astype('float32') # High res data

            yield norm(image), norm(target)  # Normalise both datasets before outputs

    def get_gen(self):
        return self.data_generator()
    

output_types = (tf.float32,tf.float32) # Specify your input and output types

patient_total =  len(patients_list) 
trai_pat_num = (int(np.rint((92/100)*patient_total))) # Select percentage of training data (rest is validation)
train_patients = patients_list[:trai_pat_num]
val_patients = patients_list[trai_pat_num:]

# Define your input and output data shape
image_size = [112,256,128,1]
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

# Define gradfient mean absolute error loss for intermediate target
class MAEGradResp(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="MAEGradResp")
  def mean_absolute_error(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)
  
  def call(self, y_true, y_pred):
    y_true = tf.transpose(y_true, (0, 2, 3, 1, 4))
    y_true = layers.TimeDistributed(layers.TimeDistributed(layers.MaxPooling1D((4))))(y_true)
    y_true = tf.transpose(y_true, (0, 3, 1, 2, 4))
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
    super().__init__(name="MAE")
  def call(self, y_true, y_pred):

    mae = tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)

    return mae

# Define mean absolute error loss for intermediate target
class CustomMAEResp(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="MAEResp")
  def call(self, y_true, y_pred):
    y_true = tf.transpose(y_true, (0, 2, 3, 1, 4))
    y_true = layers.TimeDistributed(layers.TimeDistributed(layers.MaxPooling1D((4))))(y_true)
    y_true = tf.transpose(y_true, (0, 3, 1, 2, 4))
    
    mae = tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)

    return mae

# Define custom loss that retuns zero
class ZeroLoss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__(name="ZeroLoss")
  def call(self, y_true, y_pred):
    
    return  0

# Define overall loss with weights lambda_1 and lambda_2
lambda_1 = 2.6786818474503695
lambda_2 = 0.5
sum_loss = gradient_losses.WeightedSumLoss([MAEGrad(),CustomMAE()],[lambda_1,lambda_2])
sum_loss_resp = gradient_losses.WeightedSumLoss([MAEGradResp(),CustomMAEResp()],[lambda_1,lambda_2])

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

tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=[28, 256, 128, 1])

unet = build_3d_unet_resp([28, 256, 128, 1], 2)  # Acts as a deformation field generator
deformation_fields = unet(inputs)  # Outputs the deformation fields
lambda_deformation = tf.keras.layers.Lambda(deformation)
out_2 = lambda_deformation([inputs[:, :, :, :, 0],
                            deformation_fields[:, :, :, :, 0],
                            deformation_fields[:, :, :, :, 1]])  # Outputs the deformed volume

SR_model = build_3d_unet(input_shape=(28, 256, 128, 1), num_classes=1)
SR = SR_model(out_2)
outputs = [deformation_fields, out_2, SR]
complete_model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
complete_model.compile(optimizer=optimizer, loss=[ZeroLoss(), sum_loss_resp, sum_loss])
complete_model.summary()

num_epochs = 500
model_name = 'End-to-end_model'

callbacks = [tf.keras.callbacks.ModelCheckpoint('./models_final/' + model_name, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)]

history = complete_model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, verbose=1, callbacks=callbacks)

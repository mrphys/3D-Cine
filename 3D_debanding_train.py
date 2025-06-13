## Python file for training de-banding model

# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow_mri as tfmri
import tensorflow as tf
import random
import gradient_losses
from process_utils import *

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
              target = np.load(f"./training_data/Low_Res_resp_MMWHS_{patient-hvsmr_number}.npy").astype('float32') # Data with respiratory motion
              image  = np.load(f"./training_data/Low_Res_resp_band_MMWHS_{patient-hvsmr_number}.npy").astype('float32') # Data with respiratory motion and contrast changes 

            #Load HVSMR data
            else:
              target = np.load(f"./training_data/Low_Res_resp_HVSMR_{patient}.npy").astype('float32') # Data with respiratory motion
              image  = np.load(f"./training_data/Low_Res_resp_band_HVSMR_{patient}.npy").astype('float32') # Data with respiratory motion and contrast changes 

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
    super().__init__(name="customMAEresp")
  def call(self, y_true, y_pred):
    
    mae = tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)

    return mae

# Define overall loss with weights lambda_1 and lambda_2
lambda_1 = 2.6786818474503695
lambda_2 = 0.5
sum_loss = gradient_losses.WeightedSumLoss([MAEGrad(),CustomMAE()],[lambda_1,lambda_2])

# Build and compile UNet model
tf.keras.backend.clear_session()
unet_model  = tfmri.models.UNet3D(filters=[32,64,128], kernel_size=3, out_channels=1, use_global_residual=False)
unet_model.build((None,28,256,128,1))
lr_schedule = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
unet_model.compile(optimizer=optimizer, loss=sum_loss)
unet_model.summary()


num_epochs = 500
model_name = 'De-banding_model'
callbacks=[tf.keras.callbacks.ModelCheckpoint('./models_final/' + model_name, save_best_only=True, monitor='val_loss') ,
          tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30)]

history = unet_model.fit(train_ds ,validation_data=val_ds,epochs=num_epochs, verbose=1, callbacks= callbacks)
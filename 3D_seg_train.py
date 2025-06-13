## Python file for training segmentation model

# Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from unet3plusnew import *
from process_utils import *
from volumentations import *

# Adds patient numbers for MMWHS and HVSMR data
mmwhs_number = 1 # Change to reflect number of augmented MMWHS datasets
hvsmr_number = 1 # Change to reflect number of augmented HVSMR datasets
patients_list = []
for i in range(mmwhs_number+hvsmr_number):
   patients_list.append(i)

random.Random(4).shuffle(patients_list)

class CustomDataGen():    
    def __init__(self, 
                patients,
                cohort
                ):
        self.patients = patients
        self.cohort = cohort
        
    def data_generator(self):
        for patient in self.patients:
            
            # Loads MMWHS data
            if patient >= hvsmr_number:
                target = np.load(f"./seg_data/High_Res_MMWHS_NoAug_{patient-hvsmr_number}.npy").astype('float32')
            
            # Loads HVSMR data
            else:
                target = np.load(f"./seg_data/High_Res_HVSMR_NoAug_{patient}.npy").astype('float32')

            # Does augmentations if not validation data
            if self.cohort != 'val':
                img = target[...,:1]
                lbl = target[...,1:]
                data = {'image': img, 'mask': lbl}
                aug_data = aug(**data)
                img, lbl = aug_data['image'], aug_data['mask']
                lbl = lbl > 0.2
                target = np.concatenate((img,lbl),-1)

            mask1 = target[:,:,:,1]
            mask2 = target[:,:,:,2]
            mask3 = target[:,:,:,3]
            mask4 = target[:,:,:,4]
            mask5 = target[:,:,:,5]
            mask6 = target[:,:,:,6]

            mask1 = np.clip(mask1, 0, 1)
            mask2 = np.clip(mask2, 0, 1)
            mask3 = np.clip(mask3, 0, 1)
            mask4 = np.clip(mask4, 0, 1)
            mask5 = np.clip(mask5, 0, 1)
            mask6 = np.clip(mask6, 0, 1)
            
            masks = [mask1, mask2, mask3, mask4, mask5, mask6]
            blood_pool = np.zeros_like(masks[0], dtype=np.float32)

            for mask in masks:
                blood_pool = np.logical_or(blood_pool, mask).astype(np.uint8)

            background = np.logical_not(blood_pool).astype(np.uint8)
            background = tf.expand_dims(background,-1)
            mask1 = np.expand_dims(mask1,-1).astype(np.uint8)
            mask2 = np.expand_dims(mask2,-1).astype(np.uint8)
            mask3 = np.expand_dims(mask3,-1).astype(np.uint8)
            mask4 = np.expand_dims(mask4,-1).astype(np.uint8)
            mask5 = np.expand_dims(mask5,-1).astype(np.uint8)
            mask6 = np.expand_dims(mask6,-1).astype(np.uint8)
            
            target_mask = tf.concat((background,mask1,mask2,mask3,mask4,mask5,mask6),-1)
            target_mask = np.array(target_mask).astype(np.uint8)

            target_image  = target[:,:,:,0]
            target_image = tf.expand_dims(target_image,-1)

            yield target_image, target_mask # output your inputs and outputs

    def get_gen(self):
        return self.data_generator()
    
patient_total =  len(patients_list) 
trai_pat_num = (int(np.rint((92/100)*patient_total)))
train_patients = patients_list[:trai_pat_num]
val_patients = patients_list[trai_pat_num:]

number_of_segmentations = 7 # 4 chambers, 2 vessels, background
input_shape = [112,256,128,1]
output_shape = [112,256,128,number_of_segmentations]

train_gen = CustomDataGen(train_patients, 'train').get_gen
val_gen   = CustomDataGen(val_patients, 'val').get_gen

output_signature = (tf.TensorSpec(shape=input_shape, dtype=tf.float32), 
                tf.TensorSpec(shape=output_shape, dtype=tf.uint8))

train_ds = tf.data.Dataset.from_generator(train_gen, 
                                        output_signature = output_signature)

val_ds = tf.data.Dataset.from_generator(val_gen, 
                                        output_signature = output_signature)

BATCH_SIZE = 1
train_ds = train_ds.shuffle(patient_total, seed = 42, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(-1)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(-1)

# Define focal tversky loss
def focal_tversky_loss(y_true, y_pred,alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)  # Clipping to avoid log(0)

    loss = 0.0
    num_classes = number_of_segmentations
    
    for c in range(num_classes):
        y_true_c = y_true[..., c]
        y_pred_c = y_pred[..., c]
        
        true_pos = tf.reduce_sum(y_true_c * y_pred_c)
        false_neg = tf.reduce_sum(y_true_c * (1 - y_pred_c))
        false_pos = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        
        tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
        loss_c = tf.pow((1 - tversky_index), gamma)
        loss += loss_c
    
    loss /= tf.cast(num_classes, tf.float32)  # Averaging over all classes
    return loss

# Define surface area loss loss
def SA_vol_loss(y_true,y_pred):

    loss = 0 

    y_pred = get_one_hot_tf(y_pred,number_of_segmentations)
    for i in range(number_of_segmentations-1):
        seg_true = tf.cast(y_true[...,i+1],tf.float32)
        seg_true = tf.expand_dims(seg_true,-1)
        RV_MinP_true = -tf.nn.max_pool3d(-seg_true,ksize=1,strides=1,padding='SAME')
        RV_MinP_MaxP_true = tf.nn.max_pool3d(RV_MinP_true,ksize=3,strides=1,padding='SAME')
        seg_true_SA = RV_MinP_MaxP_true - RV_MinP_true
        seg_true_SA_val = tf.reduce_sum(seg_true_SA)

        seg_pred = tf.cast(y_pred[...,i+1],tf.float32)
        seg_pred = tf.expand_dims(seg_pred,-1)
        RV_MinP_pred = -tf.nn.max_pool3d(-seg_pred,ksize=1,strides=1,padding='SAME')
        RV_MinP_MaxP_pred = tf.nn.max_pool3d(RV_MinP_pred,ksize=3,strides=1,padding='SAME')
        seg_pred_SA = RV_MinP_MaxP_pred - RV_MinP_pred
        seg_pred_SA_val = tf.reduce_sum(seg_pred_SA)

        loss = loss + tf.square((seg_true_SA_val-seg_pred_SA_val)/2500) * 10e-4

    return loss


# Create a single combined loss
def combined_loss(y_true, y_pred):
    SA = SA_vol_loss(y_true, y_pred)
    FT = focal_tversky_loss(y_true, y_pred)
    total_loss = FT +SA

    return total_loss

# Define rotation and elastic transformation augmentation
def get_augmentation():
    return Compose([
        Rotate((-10, 10), (-10, 10), (-10, 10), p=0.75),
        ElasticTransform((0, 0.1), interpolation=2, p=0.75)
    ], p=1.0)
aug = get_augmentation()


input_shape = [112,256, 128, 1]
# Instantiate the unet3plus model with the desired parameters
inputs = tf.keras.Input(shape = input_shape)
unet3 = unet3plus(inputs,
                filters=[32,64,128],
                rank = 3,  # dimension
                out_channels = number_of_segmentations,
                add_dropout = 0, # 1 or 0 to add dropout
                dropout_rate = 0.3,
                kernel_size = 3,
                encoder_block_depth= 2,
                decoder_block_depth = 1,
                pool_size = 2, # This can be either a tuple or int for same pooling across dims
                skip_type = 'encoder',
                batch_norm = 1,
                skip_batch_norm = 1,
                activation = tf.keras.layers.LeakyReLU(alpha =0.01),#'relu',
                out_activation = 'softmax',
                CGM = 0,
                deep_supervision = 0) # 1 or 0 to add deep_supervision
model = tf.keras.Model(inputs = inputs, outputs = unet3.outputs())
lr_schedule = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer,  loss=combined_loss)
model.summary()


num_epochs = 500
model_name = 'Segmentation'

callbacks=[tf.keras.callbacks.ModelCheckpoint('./models_final/' + model_name, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=60)]

history = model.fit(train_ds ,validation_data=val_ds,epochs=num_epochs, verbose=1, callbacks= callbacks)

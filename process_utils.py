import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import tensorflow_addons as tfa
from skimage.measure import label
import skimage

#Fills holes (trabeculations) in 3D mask
def fill_mask_3d(mask_3d,iter=2):
    
    pad_width = [(0,0),(20, 20), (0, 0), (0, 0),(0,0)]  # Pad 1 slice on either side of depth
    mask_padded = np.pad(mask_3d.numpy(), pad_width, mode='constant', constant_values=0)
    
    RV_MinP_true = mask_padded
    for i in range(iter):
        RV_MinP_true = tf.nn.max_pool3d(RV_MinP_true,ksize=3,strides=1,padding='SAME')
    LV_MAXP =  -RV_MinP_true   
    for i in range(iter):
        LV_MAXP = tf.nn.max_pool3d(LV_MAXP,ksize=3,strides=1,padding='SAME')
        
    filled_mask = -LV_MAXP.numpy()
    filled_mask = filled_mask[:,20:-20, :, :,:]

    return filled_mask

#Resizes image
def resize(t1,x,y):
	# Adding new axis for the channels 
	t1 = tf.expand_dims(t1, -1)

	im1 = tf.image.resize_with_crop_or_pad(t1,x,y)
	return (im1)

#Function that finds the 'Centre of Mass' of a 2D image
def find_com(image):
	if image.max() == 0:
		return image.shape[0]/2, image.shape[1]/2
	
	if math.isnan(image.max()) == True:
		return image.shape[0]/2, image.shape[1]/2

	x_list = []
	for i in range(len(image[0,:])):
		x_list.append(np.average(image[:,i])*(i))
	x_com = (np.average(x_list)/np.average(image))
	x_com=np.rint(x_com)

	y_list = []
	for i in range(len(image[:,0])):
		y_list.append(np.average(image[i,:])*(i))
	y_com = (np.average(y_list)/np.average(image))
	y_com = np.rint(y_com)

	return int(x_com), int(y_com)

#Function that normalises image
def norm(t1):
	im1= t1
	im1 = (im1-np.min(im1)) / np.max(im1)
	return (im1)

#Function that augments volumes with rotations
def rotation_augmentation(image,maxrot=np.pi*(5.0/180)):
    
    #Random seeds for rotations
	rg1 = tf.random.Generator.from_seed(1, alg='philox')
	rg2 = tf.random.Generator.from_seed(2, alg='philox')
	rg3 = tf.random.Generator.from_seed(3, alg='philox')

	# Define 3 random angles of rotation
	rotationangle_1=rg1.normal(shape=(),mean= 0, stddev=maxrot)
	rotationangle_2=rg2.normal(shape=(),mean= 0, stddev=maxrot)
	rotationangle_3=rg3.normal(shape=(),mean= 0, stddev=maxrot)
	
	#Applies first rotation
	image=tfa.image.rotate(image, angles=rotationangle_1,interpolation='bilinear',fill_mode='reflect')
	image = tf.transpose(image,(2,1,0,3))

	#Applies second rotation
	image=tfa.image.rotate(image, angles=rotationangle_2,interpolation='bilinear',fill_mode='reflect')
	image = tf.transpose(image,(1,2,0,3))
	
	#Applies third rotation
	image=tfa.image.rotate(image, angles=rotationangle_3,interpolation='bilinear',fill_mode='reflect')
	image = tf.transpose(image,(1,0,2,3))
	
	image = norm(image)

	return image

#Function that adds synthetic respiratory deformations to images
def respiratory_deformations(vol_list):
	deformed_low_res = []

	for volume in vol_list:
		def_field = []
		deformed = []
		slice_number=0

		breathing_interval = 3*np.random.random() + 3 # 3-6s per breath 
		magnitude = np.random.random()*1.75 + 0.25 # Magnitude of respiration 
		phase = np.random.random()
		heart_beat_interval = 0.6*np.random.random() +  0.6 # 0.6-1.2s per heart beat
		magnitude_2 = np.random.random()*0.5 + 0.75 #Relative streangth of AP motion to head-foot
		
		time = 0

		#Applied to each slice in the volume 
		while slice_number < np.ma.size(volume, axis = 0):
			
			#Small random variations between slices of the same volume
			variation = np.random.random()/10 +0.95 #Simulates changing breathing magnitude
			variation_2 = np.random.random()/10 +0.95 #Simulates changing heart rate
			variation_3 = np.random.random()/10 +0.95 #Simulates changing breathing rate
			
			im = volume[slice_number,:,:,0] # 2D image (slice) to be deformed
			time += 2  * heart_beat_interval * variation_2 # Time base of respiration

			#Changes breathing parameters after each breathing cycle
			if time> (breathing_interval*(slice_number+1)):

				breathing_interval = breathing_interval*variation_3
				magnitude = magnitude*variation
		
			y,x = im.shape
			dx = np.zeros((x,y)) # Define Deformation field in x
			dy = np.zeros((x,y)) # Define Deformation field in y

			x_com, y_com = find_com(im) #slice COM coordinates
   
			#Calculation of simulated deformations
			for i in range(x):
					for j in range(y):
						if j< 2*y/16 or i>(128-32) or j> 2*y_com:
							dy[i][j] = 0
							dx[i][j] = 0
						else:
							dy[i][j] = magnitude*(np.sin(2*np.pi*((time/breathing_interval)+phase)))*0.0000001*((i)*(i-96))*((j-2*y/16)*(j-2*y_com))
							dx[i][j] = -magnitude_2*magnitude*(np.sin(2*np.pi*((time/breathing_interval)+phase)))*0.00000001*((i+80)*(i-96))*((j-2*y/16)*(j-2*y_com))


			#Processing and appling deformations
			dx = np.transpose(dx)
			dy = np.transpose(dy)
			dx = tf.expand_dims(dx, axis= -1)
			dy = tf.expand_dims(dy, axis =-1) 

			im = tf.expand_dims(im,axis=-1)
			im = tf.image.convert_image_dtype(tf.expand_dims(im, 0), tf.dtypes.float32)

			displacement = tf.expand_dims(tf.concat((dy,dx), axis=-1), axis = 0)
			displacement = tf.image.convert_image_dtype(displacement, tf.dtypes.float32)

			dense_img_warp = tfa.image.dense_image_warp(im, displacement) # Applies the deformations to the image

			dense_img_warp = tf.squeeze(dense_img_warp, axis = 0)
			dense_img_warp = tf.squeeze(dense_img_warp, axis = -1)

			def_field.append(displacement)
			deformed.append(dense_img_warp)
			slice_number=slice_number+1
		
		def_field = np.array(def_field)
		def_field = def_field[:,0,...]
		deformed = np.dstack(deformed)
		deformed = np.rollaxis(deformed,-1)
		deformed_low_res.append(tf.convert_to_tensor(tf.expand_dims(norm(deformed),axis=-1)))
	return def_field,deformed_low_res

#Adds contrast differences to data
def add_bands(vol):
    band_vol = []
    for i in range(28):
        random_3 = 0.2*np.random.random()
        random = np.random.random()
        random_2 = 0.8*np.random.random() + 0.6
        if random < 0.4+random_3:
            new_slice = vol[i,:,:,:]*random_2
        else:
            new_slice = vol[i,:,:,:]
        band_vol.append(new_slice)
    final = norm(np.array(band_vol))
    return final

#One hot encode function
def get_one_hot(targets, num_classes):
    '''
    One hot encode segmentation mask
    '''
    targets = np.argmax(targets,axis = -1)
    res = np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[num_classes])

#One hot encode tensorflow function
def get_one_hot_tf(targets, num_classes):
    # Ensure targets are integers
    class_indices = tf.argmax(targets, axis=-1)  # Shape: [...], class indices along the last axis
    
    # Perform one-hot encoding on the class indices
    one_hot_encoded = tf.one_hot(class_indices, depth=num_classes)  # Shape: [..., num_classes]
    
    return one_hot_encoded

# Keeps largest component of any mask and adds residual components from other masks if forms 1 structure
def keep_largest_component_add(masks_array):
    
    
    # Extract the number of channels (i.e., masks)
    masks_number = masks_array.shape[-1]
 
    # Initialize structures
    out_masks = np.zeros_like(masks_array)  # final output array
    rest = np.zeros_like(masks_array[..., 0], dtype='float32')  # an empty mask used to accumulate leftover components not in the largest mask for each channel
    temp_large = []  # to store largest component per mask temporarily
 
   # First pass: extract largest component from each channel
    for i in range(masks_number):
        mask = masks_array[..., i]
 
        if np.sum(mask) == 0: # If mask is empty, skip to next channel
            print('channel:', i, 'mask is empty')
            largestCC = np.zeros_like(mask, dtype='float32')
            temp_large.append(largestCC)
            continue
 
        labels = label(mask) # assigns a unique int label to each connected component in the binary mask
        # print(f'labels unique in channel {i}:, {np.unique(labels)}')
        largestCC = labels == np.argmax(np.bincount(labels.flat, weights=mask.flat)) * 1 # First, counts weighted pixels per component, the finds the label with most non-zero (weighted) pixels, then returns a binary mask where only the largest component is true
        # print('labels after bincount and argmax is bool arr of shape:', (labels == np.argmax(np.bincount(labels.flat, weights=mask.flat)) * 1).shape, 'and sum:', np.sum(labels == np.argmax(np.bincount(labels.flat, weights=mask.flat)) * 1))
        largestCC = largestCC.astype('float32')
        # print('channel:', i, 'largestCC:', np.sum(largestCC), 'rest initial:', np.sum(rest), 'mask:', np.sum(mask))
        rest += mask - largestCC # add everything NOT in the largest component to the rest mask (i.e. islands not connected to the largest component).
        # print('channel:', i, 'largestCC:', np.sum(largestCC), 'rest added mask - largestCC:', np.sum(rest))
        temp_large.append(largestCC) # store the largest component of this mask for next step
 
    # Second pass: refine each mask by considering potential small components touching other structures
    for i in range(masks_number):
        if i == 0:
            out_masks[..., i] = temp_large[i].astype('int') # don't add connected components to background, just keep original mask
            continue
        if np.sum(temp_large[i]) == 0: # If mask is empty, skip to next channel
            print('channel:', i, 'temp_large is empty so not adding connected components')
            out_masks[..., i] = temp_large[i].astype('int') # don't add connected components if original channel was empty, just keep original empty mask
            continue
        combined = temp_large[i] + rest # combines current largest component with rest of the leftover pieces
        # print('\nchannel:', i, 'temp_large:', temp_large[i].sum(), 'rest:', np.sum(rest), 'combined:', np.sum(combined))
        labels = label(combined) # again applies label to find all new connected regions
        refined_largestCC = labels == np.argmax(np.bincount(labels.flat, weights=combined.flat)) # again, finds the largest connected region in this combined mask
        # print('channel:', i, 'refined_largestCC:', np.sum(refined_largestCC), 'rest:', np.sum(rest), 'temp large:', np.sum(temp_large[i]))
        refined_largestCC = refined_largestCC.astype('float32')
        rest = rest - refined_largestCC + temp_large[i]  # subtracting the new largest region removes components added to main mask from rest. Re-adding the original largest component avoids -1s in the binary mask.
        # print('channel:', i, 'refined_largestCC:', np.sum(refined_largestCC), 'rest updated:', np.sum(rest))
        out_masks[..., i] = refined_largestCC #  save the new largest component to the output list
 
    # Final pass: remove any remaining islands in the rest mask by adding to background
    out_masks[..., 0] = out_masks[..., 0] + rest # add the rest mask to the background mask
 
    # Add check that the sum of binary masks is same in input as output (i.e. each voxel belongs to only one class)
    assert masks_array.sum() == out_masks.sum(),  f"input masks sum {masks_array.sum()} should equal postprocessed masks sum {out_masks.sum()}"
 
    # output the cleaned list of masks
    return out_masks

#Applies CLAHE
def apply_clahe(img):
   
    img = (img - np.min(img))
    img = img/np.max(img)
    img = skimage.exposure.equalize_adapthist(img)
    img = (img - np.min(img))
    img = img/np.max(img)
    
    return(img)


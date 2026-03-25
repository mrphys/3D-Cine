import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import tensorflow_addons as tfa
from skimage.measure import label
import skimage
import nibabel as nib

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

def z_stand(img):
	mean = np.mean(img)
	std = np.std(img)

	# Apply z-score normalization
	z_norm_img = (img - mean) / std
	return z_norm_img

#Function that adds synthetic respiratory deformations to images
def respiratory_deformations(vol_list,xyz, scale = 1):
	deformed_low_res = []
 
	y_rand = np.random.randint(-5,5)
	z_rand = np.random.randint(-5,5)
 
	y_max = xyz[1]+y_rand
	z_max = xyz[2]+z_rand

	for volume in vol_list:
		def_field = []
		deformed = []
		slice_number=0

		breathing_interval = 3*np.random.random() + 3 # 3-6s per breath 
		magnitude = np.random.random()*1.75 + 0.10 # Magnitude of respiration 
		phase = np.random.random()
		heart_beat_interval = 0.6*np.random.random() +  0.6 # 0.6-1.2s per heart beat
		magnitude_2 = np.random.random()*0.5 + 0.75 #Relative streangth of AP motion to head-foot
		
		time = 0
		breathing_interval_check = breathing_interval

		#Applied to each slice in the volume 
		while slice_number < np.ma.size(volume, axis = 0):
			
			#Small random variations between slices of the same volume
			variation = np.random.random()/5 +0.9 #Simulates changing breathing magnitude
			variation_2 = np.random.random()/5 +0.9 #Simulates changing heart rate
			variation_3 = np.random.random()/5 +0.9 #Simulates changing breathing rate
			
			im = volume[slice_number,:,:,0] # 2D image (slice) to be deformed
			time += 2  * heart_beat_interval * variation_2 # Time base of respiration

			#Changes breathing parameters after each breathing cycle
			if time>= breathing_interval_check:
				breathing_interval = breathing_interval*variation_3
				breathing_interval_check= breathing_interval_check+ breathing_interval
				magnitude = magnitude*variation
		
			y,x = im.shape
			dx = np.zeros((x,y)) # Define Deformation field in x
			dy = np.zeros((x,y)) # Define Deformation field in y
   
			for i in range(x):
				for j in range(y):
					if j> (2*y_max+10):
						dy[i][j] = 0
						dx[i][j] = 0
					else:
						dy[i][j] = scale*magnitude*(np.sin(2*np.pi*((time/breathing_interval)+phase)))*0.00000015*((i+30)*(i-96))*((j+10)*(j-2*y_max-10))
						dx[i][j] = -scale*magnitude_2*magnitude*(np.sin(2*np.pi*((time/breathing_interval)+phase)))*0.000000025*((i+96)*(i-96))*((j+10)*(j-2*y_max-10))


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

def lowest_point_along_y(masks_nii_path):
    """
    Return (x, y, z) voxel index of the lowest point (max y index) where ANY mask channel is nonzero.
    Accepts 3D or 4D NIfTI shaped (X, Y, Z[, K]). Raises ValueError if empty.

    Parameters
    ----------
    masks_nii_path : str
        Path to the masks .nii.gz file.

    Returns
    -------
    (int, int, int)
        (x, y, z) voxel indices in array space.
    """
    nii = nib.load(masks_nii_path)
    data = np.asanyarray(nii.dataobj)
    data = data[...,:4]

    # Union across channels if 4D; otherwise treat as 3D
    if data.ndim == 4:
        union = np.any(data != 0, axis=3)
    elif data.ndim == 3:
        union = (data != 0)
    else:
        raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}")

    xs, ys, zs = np.where(union)
    if xs.size == 0:
        raise ValueError("Masks are empty (no nonzero voxels).")

    y_max = ys.max()
    pick = (ys == y_max)

    # Among voxels at max y, choose the one with largest z, then largest x
    xs_y = xs[pick]
    zs_y = zs[pick]
    order = np.lexsort((xs_y, zs_y))  # primary: z, secondary: x (ascending)
    i = order[-1]

    return int(xs_y[i]), int(y_max), int(zs_y[i])

def crop_with_padding(data, center, crop_shape):
    """
    Crops 3D or 4D data [X, Y, Z, (C)] around center with shape crop_shape, pads as needed.
    """
    data_shape = data.shape[:3]
    slices = []
    pads = []

    for i in range(3):
        half = crop_shape[i] // 2

        start = center[i] - half
        end = center[i] + crop_shape[i] - half

        pad_before = max(0, -start)
        pad_after = max(0, end - data_shape[i])

        real_start = max(0, start)
        real_end = min(end, data_shape[i])

        slices.append(slice(real_start, real_end))
        pads.append((pad_before, pad_after))

    # Crop
    if data.ndim == 4:
        cropped = data[slices[0], slices[1], slices[2], :]
        pads.append((0, 0))  # No padding for channel dimension
    else:
        cropped = data[slices[0], slices[1], slices[2]]

    # Pad as needed
    cropped_padded = np.pad(cropped, pads, mode='constant')

    return cropped_padded

# 3D-Cine

This repository contains code accompanying the *High Resolution Isotropic ‘Pseudo’ 3D Cine imaging with Automated Segmentation using Concatenated 2D Real-time Imaging and Deep Learning* paper by **Mark Wrobel**.

The pipeline is divided into three main sections:

---

## Code Structure

### 1. Training Data Pre-processing

The code uses two publicly available datasets:

- [MMWHS](https://zmiclab.github.io/zxh/0/mmwhs/)
- [HVSMR](https://segchd.csail.mit.edu/)

After downloading, place the data in the corresponding empty folders within the repository.  
Note: MMWHS requires separate folders for images and segmentations.

To pre-process the data:

1. Run `HVSMR_pre_processing.ipynb`
2. Run `MMWHS_pre_processing.ipynb`
3. Run `training_data_preprocessing.ipynb`

This will prepare all training data required for model training.

---

### 2. Model Training

Once the data is pre-processed, train the deep learning models by running the following scripts:

- `3D_contrast_correction_train.py`
- `3D_respcor_train.py`
- `3D_E2E_train.py`
- `3D_seg_train.py`

Make sure to update the `mmwhs_number` and `hvsmr_number` variables to reflect the number of processed datasets.

---

### 3. Inference and Post-processing

There are two .exe files in the **installer** folder. These install a local app for Windows to run the image correction models or the segmentation model. Both can run on CPU (slow) or DirectML (GPU acceleration) if available. The 3D Cine app expects a .zip file of the real-time concatenated sagittal 2D stack and ouptuts a .zip of the processed DICOM data. The processed .zip can then be dropped straight into the 3D Cine Segmentation app. Note: The segmentation model requires around 16Gb GPU RAM to use DirectML 

---

## Docker Support

A `Dockerfile` is included for creating a reproducible environment.

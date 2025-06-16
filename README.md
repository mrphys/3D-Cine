# 3D-Cine

This repository contains code accompanying the *3D Cine* paper by **Mark Wrobel**.

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

- `3D_debanding_train.py`
- `3D_respcor_train.py`
- `3D_E2E_train.py`
- `3D_seg_train.py`

Make sure to update the `mmwhs_number` and `hvsmr_number` variables to reflect the number of processed datasets.

---

### 3. Inference and Post-processing

Run `3D_cine_post_processing.ipynb`.

This step can be run independently of the training pipeline.  
An example healthy volunteer dataset is located in the `raw_data` folder and is processed using trained models located in `models_final/`.

Output 3D Cine data and segmentations are saved as `.npy` arrays in the `processed_data` folder.

---

## Docker Support

A `Dockerfile` is included for creating a reproducible environment.

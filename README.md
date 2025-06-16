# 3D-Cine

This repository contains code accompanying the *3D Cine* paper by **Mark Wrobel**.

The pipeline is divided into three main sections:

---

## 🔧 Code Structure

### 1. **Training Data Pre-processing**
- The code uses two publicly available datasets:
  - [MMWHS](https://zmiclab.github.io/zxh/0/mmwhs/)
  - [HVSMR](https://segchd.csail.mit.edu/)
- After downloading, place the data in the corresponding empty folders within the repository.
  - ⚠️ *Note:* MMWHS requires separate folders for images and segmentations.
- Pre-processing steps:
  1. Run `HVSMR_pre_processing.ipynb`
  2. Run `MMWHS_pre_processing.ipynb`
  3. Run `training_data_preprocessing.ipynb`
- This will prepare all training data required for model training.

---

### 2. **Model Training**
- Once data is pre-processed, train the deep learning models by running the following scripts:
  - `3D_debanding_train.py`
  - `3D_respcor_train.py`
  - `3D_E2E_train.py`
  - `3D_seg_train.py`
- 📌 Make sure to update the variables `mmwhs_number` and `hvsmr_number` to match the number of pre-processed datasets.

---

### 3. **Inference & Post-processing**
- Run the notebook: `3D_cine_post_processing.ipynb`
- This step requires no prior scripts to be run.
- An example healthy volunteer dataset is included in the `raw_data` folder.
- The models used for inference are located in `models_final/`.
- The output:
  - 3D Cine reconstructions and segmentations
  - Saved as `.npy` arrays in the `processed_data` folder

---

## 📦 Docker Support

A `Dockerfile` is provided for setting up a reproducible environment.

---

## 📁 Folder Overview

```bash
├── raw_data/            # Input data (e.g. example volunteer cine stack)
├── processed_data/      # Output segmentations and 3D cine volumes
├── models_final/        # Trained model checkpoints
├── Dockerfile           # For containerized execution
├── *.py, *.ipynb        # Scripts and notebooks for preprocessing, training, and inference
└── README.md

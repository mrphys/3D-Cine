Code as described in 3D Cine paper by Mark Wrobel.

There are 3 distinct sections of the code: i) Training data pre-processing files, ii) Training data files and iii) 3D cine inference and post-processing with trained models and example volunteer dataset.
A Dockerfile is also included for convenience.

i): The training data used here is open-source MMWHS (https://zmiclab.github.io/zxh/0/mmwhs/) and HVSMR (https://segchd.csail.mit.edu/) datasets. Once downloaded, the data should be placed in their corresponding empty folders (note MMWHS has seperate image and segmentation folders). Now the HVSMR_pre_processing and MMWHS_pre_processing notebooks can be run, followed by the training_data_preprocessing notebook. All the training data is now ready.

ii): Once the training data has been processed, the four Deep Learning models can be trained by running 3D_debanding_train, 3D_respcor_train, 3D_E2E_train and 3D_seg_train .py files (take care to change the mmwhs_number and hvsmr_number variables to match the total number of pre_processed datasets.)

iii): The 3D_cine_post_processing notebook can be run without running any other files. An example healthy volunteer dataset is located in the raw_data folder and is processed with trained models located in models_final. Output 3D Cine data and corresponding segmentations are saved as .npy arrays in the processed_data folder.
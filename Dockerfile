# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.0/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
FROM tensorflow/tensorflow:2.9.1-gpu
RUN apt-get install libopenexr-dev -y
RUN pip install tensorflow-mri
RUN pip install tqdm
RUN pip install h5py
RUN pip install tensorflow-addons
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install neptune-client
RUN pip install matplotlib
RUN pip install scipy
RUN pip install pydicom
RUN pip install nibabel 
RUN pip uninstall -y typing_extensions && \
    pip install typing_extensions==4.11.0
RUN pip install volumentations


# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1003
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to sudoers.
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Change default shell to bash.
    usermod --shell /bin/bash $USERNAME
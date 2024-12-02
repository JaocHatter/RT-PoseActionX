<h1 align="center">PoseActionX: Real-time Skeleton-Based Action Recognition with Graph Convolutional Networks</h1>

<p align="center">
  <!-- Add badges here if available -->
</p>

Official PyTorch implementation aiming to provide multiple skeleton-based human action recognition models for real-time applications using OpenCV and MediaPipe. 

# Overview

PoseActionX is a project focused on implementing various skeleton-based action recognition models for real-time use cases. The framework is designed to support multiple models, enabling users to train and deploy action recognition systems efficiently. While HDGCN is the only model implemented at the moment, the project structure allows for easy integration of additional models in the future.

## UCF101 Videos as Demo
![user_demo_0](https://github.com/user-attachments/assets/88993c17-abff-49a2-9349-c4a6b88e2949)
![user_demo_1](https://github.com/user-attachments/assets/211def57-4d60-4196-9067-083bf90022f9)
![user_demo_2](https://github.com/user-attachments/assets/15b6b567-db52-4ae0-8a76-4d3695d2ce76)


# Installation

## Prerequisites

- **Python** >= 3.6
- **Rust** >= 1.82.0 (Required if you are going to use Joint Extractor tool)
- **PyTorch** (with CUDA support recommended for GPU acceleration)
- **Additional Python Packages**:

  - `torch`
  - `einops`
  - `tqdm` >= 4.66.5
  - `numpy` >= 1.26.3
  - `scikit-learn` >= 1.5.2
  - `opencv-python` >= 4.10.0.84
  - `mediapipe` >= 0.10.15

All required packages are specified in the `setup.py` and can be installed using the provided `Makefile`.

## Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/poseactionx.git
   cd poseactionx
   ```

2. **Install Dependencies**

   The dependencies are specified in `setup.py`. Install them by running:

   ```bash
   make install
   ```

   This command will install all required packages.

   **Note:** Ensure that PyTorch is installed with CUDA support if you plan to use GPU acceleration during training or inference.

3. **Verify Installation**

   After installation, verify that the `poseactionx` command-line tool is available:

   ```bash
   poseactionx --help
   ```

# Project Structure

The project has the following structure:

```
.
├── config
│   └── hdgcn_conf.yaml
├── demo
│   └── user_demo.mp4
├── JointExtractor
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── output_dataset
│   ├── pose_extractor.py
│   ├── README.md
│   ├── src
│   │   └── main.rs
├── LICENSE
├── main.py
├── Makefile
├── models
│   ├── graphs
│   │   ├── graph.py
│   │   ├── __init__.py
│   │   └── tools.py
│   ├── HDGCN.py
│   ├── __init__.py
│   └── load.py
├── output
│   ├── hdgcn_model_epoch_12_beta.pt
│   ├── hdgcn_model_epoch_34_beta.pt
│   └── hdgcn_model_epoch_7_beta.pt
├── save_demo.py
├── setup.py
└── training_functions.py
```

- **config/**: Contains configuration files for different models.
- **demo/**: Directory for demo scripts and saved models.
- **JointExtractor/**: Tools and scripts for extracting joint data from datasets.
- **models/**: Implementation of models and graph structures.
- **Makefile**: Contains commands for installation and running the project.
- **setup.py**: Script for installing dependencies.
- **main.py**: Main entry point for training and testing models.

# Data Preparation

To train the model, you need to prepare your dataset in the required format.

1. **Data Files**

   - `final_joints.npy`: NumPy array containing the joint data.
   - `final_labels.npy`: NumPy array containing the corresponding labels.

2. **Directory Structure**

   Place these files in the following directory:

   ```
   JointExtractor/output_dataset/
   ```

3. **Data Format Details**

   For detailed information about the data format and how to generate these files, please refer to the `README.md` in the `JointExtractor` folder.

# Training & Testing

## Training

To train the model, use the following command:

```bash
make run ARGS=hdgcn
```

- `ARGS=hdgcn` specifies the model to use for training. Currently, `hdgcn` (Hierarchically Decomposed Graph Convolutional Network) is implemented.
- More models will be added in the future and can be specified similarly.

**Note:** Ensure your data is correctly prepared and placed in the `JointExtractor/output_dataset` directory before starting training.

## Model Saving

After training, the model weights will be saved in the directory:

```
demo/saved_models/
```

## Real-time Testing

You can test the trained model in real time using OpenCV and MediaPipe.

### Steps to Run Real-time Demo

1. **Ensure Webcam Access**

   Make sure your system has a webcam connected and accessible.

2. **Run the Real-time Demo Script**

   ```bash
   python save_demo.py
   ```

   This script will:

   - Capture video from your webcam using OpenCV.
   - Use MediaPipe to perform real-time pose estimation.
   - Utilize the trained model to recognize actions based on the skeleton data.
   - Display the action predictions on the video feed.

3. **Adjust Parameters**

   If you need to specify the path to the saved model or adjust other parameters, you can modify the script or pass arguments as needed.

### Demo Videos and Images

Demo videos and images showcasing the real-time action recognition capabilities will be added to the `demo` directory in future updates.

# GPU Support

To leverage GPU acceleration during training and inference:

- Ensure that PyTorch is installed with CUDA support.
- Verify that your system has a compatible NVIDIA GPU and the appropriate CUDA drivers installed.

# Acknowledgements

This project is based on the implementation of [Hierarchically Decomposed Graph Convolutional Networks (HDGCN)](https://github.com/Jho-Yonsei/HD-GCN) for skeleton-based action recognition.

Special thanks to the original authors for their valuable work!

# Citation

If you find this project useful in your research, please consider citing the models authors, and of course this repo ;)

```bibtex
@InProceedings{Lee_2023_ICCV,
    author    = {Lee, Jungho and Lee, Minhyeok and Lee, Dogyoon and Lee, Sangyoun},
    title     = {Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10444-10453}
}
```

# Future Work

- **Additional Models**: Support for more skeleton-based action recognition models will be added in future updates (spoiler: DeGCN).
- **Demo Videos**: We plan to include demo videos and images showcasing the real-time action recognition capabilities in the `demo` directory.


# Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainer at [jarex1012@gmail.com].

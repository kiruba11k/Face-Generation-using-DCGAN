# Face Generation using DCGAN

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic images of human faces. It uses the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for training, pre-processed to contain cropped and resized images.

## Features
- Train a DCGAN to generate realistic human face images.
- Visualize generated faces at different training stages.
- Utilize PyTorch's `DataLoader` and `ImageFolder` for efficient data handling.

---

## Prerequisites
1. **Python** (>= 3.7)
2. **Pip** package manager
3. **PyTorch** (>= 1.10) with CUDA for GPU acceleration
4. **Additional Python packages**:
   - `torchvision`
   - `matplotlib`
   - `numpy`

---

## Setup Guide

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/kiruba11k/Face-Generation-using-DCGAN.git
cd Face-Generation-using-DCGAN
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision matplotlib numpy
```

### 3. Download the Dataset
Download the pre-processed CelebA dataset:
- Visit [this link](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip).
- Extract the downloaded ZIP file into the project folder.

Ensure the directory structure is as follows:
```
project_folder/
├── Face-Generation-using-DCGAN/
│   ├── Dataset
|   |--- main.py

```

### 4. Verify Installation
Run the following command to verify everything is set up correctly:
```bash
python dlnd_face_generation.ipynb
```

### 5. Train the Model
Follow these steps in the Jupyter Notebook:
1. Load the pre-processed dataset using the `get_dataloader` function.
2. Define the generator and discriminator architectures.
3. Train the DCGAN by running the notebook cells.
4. Visualize generated faces after training.

---

## Project Components
1. **Dataset Preprocessing**:
   - Resize images to `64x64` resolution.
   - Normalize and batch data using `PyTorch` utilities.

2. **DCGAN Implementation**:
   - Define the generator to create realistic images.
   - Build the discriminator to distinguish real vs. fake images.

3. **Training**:
   - Train adversarial networks with a defined loss function.
   - Monitor the quality of generated images during training.

4. **Visualization**:
   - Visualize both training progress and generated images using `matplotlib`.

---

## Results
Generated faces after sufficient training resemble realistic human faces, with only minor noise.

---

## Troubleshooting
- **GPU Requirement**: Ensure a CUDA-compatible GPU is available for faster training.
- **Out of Memory**: Reduce the batch size if you encounter memory issues.

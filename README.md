# Actor Resemblance Classifier using K-Nearest Neighbors (KNN)

This repository demonstrates a Computer Vision application that identifies actor resemblance using a K-Nearest Neighbors (KNN) algorithm implemented from scratch. By treating images as high-dimensional vectors, the system compares a user-provided test image against a labeled dataset to find the most statistically similar matches.

## Overview

The system processes image data through a complete machine learning pipeline in Google Colab:
- **Data Acquisition**: Loading labeled actor data from MATLAB `.mat` files.
- **Preprocessing**: Normalizing pixel intensities and performing multidimensional array reshaping.
- **Metric Computation**: Implementing Euclidean distance calculations across 3072-dimensional feature vectors.
- **Classification**: Implementing 1NN, 3NN, and 5NN logic with custom majority voting.

## Engineering Logic: Image-Based KNN

### 1. Data Structuring & Reshaping
The dataset consists of 50 color images ($32 \times 32 \times 3$). To facilitate distance calculations, the images are reshaped into 1D vectors of size 3072 using **Column-major (Fortran-style) ordering** to preserve spatial integrity:
- `images = np.reshape(images, (32, 32, 3, -1), order="F")`.

### 2. Distance Metric
The core resemblance is determined by calculating the **Euclidean Distance** between the test image vector ($T$) and each dataset image vector ($G$):

$$dist = \sum (T - G)^2$$

This provides a numerical value representing the "dissimilarity" between two faces.

### 3. Majority Voting (k=3, k=5)
For higher values of $k$, the system identifies the $k$ smallest distances and uses a **Majority Voting** scheme to determine the final label, mitigating the impact of outliers or lighting variations in a single image.


## Tech Stack

- **Environment:** Google Colab / Jupyter Notebook 
- **Language:** Python 3
- **Libraries:**
    - **OpenCV (cv2):** Image resizing and interpolation.
    - **NumPy:** Vectorized matrix operations and distance math.
    - **Matplotlib:** Visualizing image results and RGB channels.
    - **SciPy:** Handling `.mat` dataset structures.

## 📂 Project Structure

- **Actor_Resemblance_KNN.ipynb**: The complete documented pipeline from data loading to 5NN classification.
- **Project Data**: Uses a dataset of 50 images across 10 classes (5 images per actor) including labels for Meryl Streep, Morgan Freeman, Viola Davis, and others.

## Results

The model successfully identifies resemblances by displaying the closest matching index and its associated label from the dataset. For example, a successful 1NN match identifies the minimum distance at a specific dataset index (e.g., Index 4: Morgan Freeman).

<img width="993" height="382" alt="Screenshot 2026-01-01 at 5 50 09 PM" src="https://github.com/user-attachments/assets/6f80ecff-9747-4ba8-ae7c-c21a7e914488" />


## ⚠️ Academic Integrity

This repository is intended solely as a piece to showcase my learning journey. 
If you are a student working on a similar assignment: **do not copy this code.** Plagiarism is a serious offense that can lead to expulsion. Use this only as a conceptual reference to understand KNN and high-dimensional image processing.

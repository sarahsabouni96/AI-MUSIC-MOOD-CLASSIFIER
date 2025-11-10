# AI Music Mood Classifier

## Overview
The **AI Music Mood Classifier** is a machine learning project that predicts the mood of a song based on its audio features. Using a neural network implemented in PyTorch, the model classifies songs into one of four moods: **sad**, **happy**, **energetic**, or **calm**.

The classifier uses song features such as tempo, danceability, speechiness, liveness, and more to predict the mood of a song.

---

## Dataset & Features

The dataset consists of numerical audio features extracted from songs. Key features include:

- danceability  
- energy  
- speechiness  
- acousticness  
- instrumentalness *(dropped during preprocessing)*  
- liveness  
- valence  
- tempo  
- duration_min *(converted from milliseconds)*
    
---

## Dataset Source

The dataset used in this project is from Kaggle: [Moodify Dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset?resource=download). and contains numerical audio features extracted from songs.

---

## Target Labels
The target variable is the song mood, represented as integer labels.

| Label | Mood      |
|-------|-----------|
| 0     | sad       |
| 1     | happy     |
| 2     | energetic |
| 3     | calm      |

These numeric labels correspond to the categorical moods, which the neural network predicts using a multi-class classification approach.

---

## Data Preprocessing

Before training the model, the following preprocessing steps were applied:

- **Duplicate removal** – to avoid repeated data  
- **Log-transform** – applied to skewed features such as `duration_min`, `speechiness`, `liveness`, and `tempo`  
- **Outlier detection** – using IQR and modified z-score methods  
- **Multi-column outlier filtering** – removed rows that were outliers in 2+ features  
- **Winsorization** – capped extreme values at the 1st and 99th percentiles  
- **Scaling** – features were standardized using `StandardScaler`

---

## Model Architecture

The neural network is implemented in PyTorch with the following structure:

- **Input layer:** 9 nodes  
- **Hidden Layer 1:** 64 neurons, ReLU activation, 30% dropout  
- **Hidden Layer 2:** 32 neurons, ReLU activation, 30% dropout  
- **Output layer:** 4 neurons (raw logits for 4 moods)  

**Optimizer:** Adam (learning rate = 0.001)  
**Loss function:** CrossEntropyLoss  
**Training:** Mini-batch gradient descent, batch size = 64, 30 epochs  

---

## Results

- **Test Accuracy:** ~80%  
- **Loss curves:** Training and test loss monitored across epochs  
- **Confusion Matrix:** Displays classification performance per class  

---

## Visualizations

All relevant plots, including feature histograms, correlation matrices, outlier checks, and loss curves, are included in the repository for detailed inspection.

---

## Usage

To run the project:

1. Install required Python packages:
2. Place `278k_labelled_uri.csv` in the project directory.
3. Run the Jupyter notebook or Python script to train and evaluate the model.

---


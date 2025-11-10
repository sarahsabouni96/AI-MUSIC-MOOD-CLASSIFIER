# AI-MUSIC-MOOD-CLASSIFIER
Overview

The AI Music Mood Classifier is a machine learning project that predicts the mood of a song based on its audio features. Using a neural network implemented in PyTorch, the model classifies songs into one of four moods: sad, happy, energetic, or calm.

The classifier uses song features such as tempo, danceability, speechiness, liveness, and more to predict the mood of a song.

Dataset & Features

The dataset consists of numerical audio features extracted from songs. Key features include:

danceability

energy

speechiness

acousticness

instrumentalness (dropped during preprocessing)

liveness

valence

tempo

duration_min (converted from milliseconds)

The target variable is the song mood, represented as integer labels.

Target Labels

The model predicts the following moods:

Label	Mood
0	sad
1	happy
2	energetic
3	calm

These numeric labels correspond to the categorical moods, which the neural network predicts using a multi-class classification approach.

Data Preprocessing

Before training the model, the following preprocessing steps were applied:

Duplicate removal – to avoid repeated data.

Log-transform – applied to skewed features such as duration_min, speechiness, liveness, and tempo to reduce skewness.

Outlier detection – using IQR and modified z-score methods. Multi-column outliers (rows that were outliers in 2+ features) were removed.

Winsorization – extreme values were capped at the 1st and 99th percentiles to limit the influence of outliers without removing rows.

Scaling – features were standardized using StandardScaler to have zero mean and unit variance.

Model Architecture

The neural network is implemented in PyTorch with the following structure:

Input layer: 9 nodes (corresponding to the input features)

Hidden Layer 1: 64 neurons, ReLU activation, 30% dropout

Hidden Layer 2: 32 neurons, ReLU activation, 30% dropout

Output layer: 4 neurons (corresponding to 4 moods), raw logits fed into CrossEntropyLoss

Optimizer: Adam (learning rate = 0.001)
Loss function: Cross-entropy loss (suitable for multi-class classification)
Training: Mini-batch gradient descent, batch size = 64, 30 epochs

Results

Training and Test Loss: Monitored across epochs

Test Accuracy: ~80%

Confusion Matrix: Shows how well the model predicts each mood

Usage

To run the project:

Install required Python packages (pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, torch, tqdm).

Place songs.csv in the project directory.

Run the Jupyter notebook or Python script to train the model and evaluate results.


#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # for progress bar

#%% Step 1: Load dataset
df = pd.read_csv('278k_labelled_uri.csv') 
df = df.drop(df.columns[:2], axis=1)  # Drop ID/URI

print("Initial data:")
print(df.head())

# Histogram for each feature 
df.hist(figsize=(10, 8))
plt.show()

print(df.columns)

# Correlation 
numeric_features = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_features.corr()
print(corr_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.show()


#%% Step 2: Cleaning
cols_to_drop = ['instrumentalness', 'loudness', 'spec_rate', 'uri']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# Duration in minutes
df_clean['duration_min'] = df_clean['duration (ms)'] / 60000
df_clean = df_clean.drop(columns=['duration (ms)'], errors='ignore')

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"Remaining rows after duplicates removed: {len(df_clean)}")

#%% Step 3: Visual check for outliers
numeric_features = df_clean.select_dtypes(include=['float64', 'int64']).copy()

# Remove 'labels' from numeric features if present
if 'labels' in numeric_features.columns:
    numeric_features = numeric_features.drop(columns=['labels'])

# Boxplots without labels
for col in numeric_features.columns:
    plt.figure(figsize=(6,4))
    plt.boxplot(df_clean[col], vert=False)
    plt.title(f"Outliers in {col}")
    plt.xlabel(col)
    plt.show()


#%% Step 4: Log-transform skewed columns
log_cols = ['duration_min', 'speechiness', 'liveness', 'tempo']
for col in log_cols:
    df_clean[col] = np.log1p(df_clean[col])

#%% Step 5: Outlier detection (IQR + modified z-score)
def modified_z_score(series):
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad == 0:
        std = series.std()
        return (series - series.mean()) / (std if std != 0 else 1.0)
    return 0.6745 * (series - med) / mad

numeric_df = df_clean.select_dtypes(include=['float64', 'int64']).copy()
if 'labels' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['labels'])

iqr_threshold = 1.5
modz_threshold = 3.5
outlier_indices_per_col = {}

for col in numeric_df.columns:
    s = numeric_df[col].dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    if IQR > 0:
        lower = Q1 - iqr_threshold * IQR
        upper = Q3 + iqr_threshold * IQR
        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
    else:
        mz = modified_z_score(numeric_df[col].fillna(numeric_df[col].median()))
        mask = mz.abs() > modz_threshold

    outlier_indices_per_col[col] = numeric_df.index[mask].tolist()

#%% Step 6: Multi-column outlier filtering (2+ columns)
row_outlier_counts = pd.Series(0, index=numeric_df.index)
for col, indices in outlier_indices_per_col.items():
    row_outlier_counts[indices] += 1

multi_col_outliers = row_outlier_counts[row_outlier_counts >= 2].index
print(f"Rows flagged as outliers in 2+ columns: {len(multi_col_outliers)}")

df_filtered = df_clean.drop(index=multi_col_outliers)
print(f"Remaining rows after multi-column outlier filter: {len(df_filtered)}")

#%% Step 7: Winsorization
numeric_cols_filtered = df_filtered.select_dtypes(include=['float64', 'int64']).columns
df_winsorized = df_filtered.copy()
for col in numeric_cols_filtered:
    if df_winsorized[col].dropna().nunique() >= 3:
        df_winsorized[col] = mstats.winsorize(df_winsorized[col], limits=[0.01, 0.01])

#%% Step 8: Prepare inputs for PyTorch
X = df_winsorized.drop(columns=['labels']).values.astype(np.float32)
y = df_winsorized['labels'].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
X_test_t = torch.tensor(X_test)
y_train_t = torch.tensor(y_train)
y_test_t = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#%% Step 9: Define neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size=9, hidden1=64, hidden2=32, output_size=4, dropout=0.3):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # logits
        return x

#%% Step 10: Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_t = X_train_t.to(device)
X_test_t = X_test_t.to(device)
y_train_t = y_train_t.to(device)
y_test_t = y_test_t.to(device)

#%% Step 11: Training loop with mini-batches and progress bar
num_epochs = 30
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    
    train_losses.append(epoch_loss / len(train_loader.dataset))
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t)
        test_losses.append(test_loss.item())
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

#%% Step 12: Plot losses
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

#%% Step 13: Predictions and accuracy
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_t)
    y_pred = torch.argmax(y_pred_probs, dim=1)

acc = accuracy_score(y_test_t.cpu(), y_pred.cpu())
print(f"Test Accuracy: {acc*100:.2f}%")

#%% Step 14: Confusion matrix
cm = confusion_matrix(y_test_t.cpu(), y_pred.cpu())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
#%%
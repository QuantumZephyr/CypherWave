import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, Callback
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
import multiprocessing

# Read Excel file
data = pd.read_excel('F:/Activation energies/data20240525.xlsx')

# Extract feature and target columns
features = data[['K1', 'K2', 'K3', 'K4', 'K5', 'KInf', 'K_1', 'K_2', 'K_3', 'K_4', 'K_5', 'K_Inf', 'DeltH']]
target = data['Activation_energies']

# Process Reactant's Morgan Fingerprint
smiles_fingerprints = data['Reactant_Fingerprint'].apply(lambda x: [int(c) for c in str(x)])

# Convert to DataFrame
smiles_fingerprint_df = pd.DataFrame(smiles_fingerprints.tolist(), columns=['smiles_fingerprints_' + str(i) for i in range(len(smiles_fingerprints.iloc[0]))])

# Process Product's Morgan Fingerprint
Product_fingerprints = data['Product_Fingerprint'].apply(lambda x: [int(c) for c in str(x)])

# Convert to DataFrame
Product_fingerprint_df = pd.DataFrame(Product_fingerprints.tolist(), columns=['Product_fingerprints_' + str(i) for i in range(len(Product_fingerprints.iloc[0]))])

# Merge processed Morgan Fingerprint columns with other feature columns
features = pd.concat([features, smiles_fingerprint_df, Product_fingerprint_df], axis=1)

# Create StandardScaler objects
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Normalize data
scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1)).flatten()

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Split validation set and remaining dataset
total_size = len(scaled_features)
val_size = int(0.1 * total_size)  # Validation set is 10% of the total data
all_indices = np.arange(total_size)
np.random.shuffle(all_indices)

val_indices = all_indices[:val_size]
train_test_indices = all_indices[val_size:]

X_val, y_val = scaled_features[val_indices], scaled_target[val_indices]
X_train_test, y_train_test = scaled_features[train_test_indices], scaled_target[train_test_indices]

# Create KFold object
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize performance metrics lists
training_r2_scores = []  # Store training set R2 scores
training_mae_scores = []
training_rmse_scores = []

r2_scores = []
mae_scores = []
rmse_scores = []

# Define the artificial neural network model
model = Sequential()
model.add(Dense(500, activation='elu', kernel_initializer='he_normal', input_dim=scaled_features.shape[1]))
model.add(Dropout(0.05))
model.add(Dense(250, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(125, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(62, activation='elu', kernel_initializer='he_normal'))
model.add(Dropout(0.05))
model.add(Dense(1))

# Compile the model using SGD optimizer
model.compile(loss='mean_squared_error', optimizer='adamax')

# Define Early Stopping callback function
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = {'batch': [], 'epoch': []}
        self.val_losses = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs=None):
        self.losses['batch'].append(logs.get('loss'))
        if 'val_loss' in logs:
            self.val_losses['batch'].append(logs.get('val_loss'))
        else:
            self.val_losses['batch'].append(None)

    def on_epoch_end(self, epoch, logs=None):
        self.losses['epoch'].append(logs.get('loss'))
        if 'val_loss' in logs:
            self.val_losses['epoch'].append(logs.get('val_loss'))
        else:
            self.val_losses['epoch'].append(None)

# Initialize lists to save all training and validation losses
all_train_losses = []
all_val_losses = []
for _ in range(8):
    for i, (train_indices, test_indices) in enumerate(kf.split(X_train_test)):
        # Split training and test sets
        X_train, y_train = X_train_test[train_indices], y_train_test[train_indices]
        X_test, y_test = X_train_test[test_indices], y_train_test[test_indices]

        # Initialize LossHistory object
        loss_history = LossHistory()

        # Train model (using callback functions)
        history = model.fit(X_train, y_train,
                            epochs=100,
                            batch_size=32,
                            callbacks=[early_stopping, loss_history],
                            validation_data=(X_test, y_test),
                            verbose=0)
        # Save each fold's training and validation losses
        all_train_losses.extend(loss_history.losses['epoch'])
        all_val_losses.extend(loss_history.val_losses['epoch'])

        # Predict training set
        y_train_pred = model.predict(X_train).flatten()

        # Predict test set
        predictions = model.predict(X_test).flatten()

        # Calculate training set R2 score and add to list
        training_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        training_r2_scores.append(training_r2)
        training_mae_scores.append(train_mae)
        training_rmse_scores.append(train_rmse)

        # Calculate test set R2 score and add to list
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        test_mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(test_mse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

# Calculate average performance metrics for training set
mean_training_r2 = sum(training_r2_scores) / len(training_r2_scores)
mean_train_mae = sum(training_mae_scores) / len(training_mae_scores)
mean_train_rmse = sum(training_rmse_scores) / len(training_rmse_scores)

# Calculate average performance metrics for test set
mean_r2 = sum(r2_scores) / len(r2_scores)
mean_mae = sum(mae_scores) / len(mae_scores)
mean_rmse = sum(rmse_scores) / len(rmse_scores)

# Print average performance metrics for training set
print("Average train R2 Score:", mean_training_r2)
print("Average train MAE:", mean_train_mae)
print("Average train RMSE:", mean_train_rmse)

# Print average performance metrics for test set
print("Average Test R2 Score:", mean_r2)
print("Average Test MAE:", mean_mae)
print("Average Test RMSE:", mean_rmse)

# Plot scatter plot for training and test sets
plt.figure(figsize=(10, 8))
# Training set scatter plot
plt.scatter(y_train, y_train_pred, color='purple', label='Train Set', alpha=0.6)
# Test set scatter plot
plt.scatter(y_test, predictions, color='blue', label='Test Set', alpha=0.6)
# Plot reference line y=x
max_value = max(max(y_train), max(y_test), max(y_train_pred), max(predictions))
min_value = min(min(y_train), min(y_test), min(y_train_pred), min(predictions))
plt.plot([min_value, max_value], [min_value, max_value], color='orange', linestyle='--', label='y=x')
# Add titles and labels
plt.xlabel('Actual Values', fontsize=16)
plt.ylabel('Predicted Values', fontsize=16)
plt.title('Actual vs Predicted Values', fontsize=18)
plt.legend()
# Save plot
plt.savefig('F:/Activation energies/scatter_xiugzheng_GPT_val_inside_scaled_1.png', format='png', dpi=300, bbox_inches='tight')
# Show plot
plt.show()

# Plot learning curve
plt.figure(figsize=(10, 8))
plt.plot(all_train_losses, label='Training loss', color='purple')
plt.plot(all_val_losses, label='Validation loss', color='orange')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Learning Curve', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig('F:/Activation energies/learning_curve_xiugzheng_GPT_val_inside_scaled.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
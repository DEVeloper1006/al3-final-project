# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
# from keras.regularizers import l2
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from data_load import get_data
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# X_train, X_test, Y_train, Y_test, mapping = get_data("traffic_data.parquet", ' Label',[' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'])
# print(mapping)

# # Reshape the data for CNN (add a channel dimension)
# X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# # Build CNN model
# def build_cnn(input_shape):
#     model = Sequential()

#     # First convolutional layer
#     model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
#                      kernel_regularizer=l2(0.001), input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.25))

#     # Second convolutional layer
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.25))

#     # Fully connected layer
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))

#     # Output layer
#     model.add(Dense(12, activation='softmax'))  # Adjusted for multi-class classification
    
#     return model

# input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
# model = build_cnn(input_shape)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     X_train, Y_train, 
#     validation_data=(X_test_cnn, Y_test),
#     epochs=10, 
#     batch_size=32, 
#     callbacks=[early_stopping, lr_scheduler]
# )

# # Evaluate on the test set
# test_loss, test_accuracy = model.evaluate(X_test_cnn, Y_test)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# # Predict the class probabilities for the test set
# y_pred_probs = model.predict(X_test_cnn)

# # Convert probabilities to class predictions
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Compute the confusion matrix
# conf_matrix = confusion_matrix(Y_test, y_pred)

# # Display the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=mapping.values(), yticklabels=mapping.values())
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# # Optional: Print classification report for more insights
# print(classification_report(Y_test, y_pred, target_names=mapping.values()))

# weights_file = 'cnn_model_weights.h5'
# model.save_weights(weights_file)

# print(f"Model weights saved to {weights_file}")

import tensorflow as tf
import setuptools.dist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler  # For oversampling
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from data_load import get_data
import numpy as np

# Load data
X_train, X_test, Y_train, Y_test, mapping = get_data(
    "traffic_data.parquet",
    ' Label',
    [' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
     ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', 
     ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 
     ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', 
     ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', 
     ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', 
     ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', 
     ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', 
     ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', 
     ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', 
     ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', 
     ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', 
     ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 
     'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', 
     ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 
     ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', 
     ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
)

X_train = X_train[:500000, :]
Y_train = Y_train[:500000]
X_test = X_test[:10000, :]
Y_test = Y_test[:10000]
print(mapping)

# Reshape the data for CNN (add a channel dimension)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Sampling Strategy
def apply_sampling(X, Y):
    # Reshape for sampling (flatten the data temporarily)
    X_flattened = X.reshape(X.shape[0], -1)
    # Apply Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, Y_resampled = ros.fit_resample(X_flattened, Y)
    # Reshape back to CNN-compatible shape
    X_resampled = X_resampled.reshape(X_resampled.shape[0], X.shape[1], X.shape[2])
    # Shuffle to randomize the data order
    X_resampled, Y_resampled = shuffle(X_resampled, Y_resampled, random_state=42)
    return X_resampled, Y_resampled

# Apply the sampling strategy to the training data
X_train_resampled, Y_train_resampled = apply_sampling(X_train_cnn, Y_train)

# Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Build CNN model
def build_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
                     kernel_regularizer=l2(0.001), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))  # Adjusted for multi-class classification
    return model

input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
model = build_cnn(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN
history = model.fit(
    X_train_resampled, Y_train_resampled,
    validation_data=(X_test_cnn, Y_test),
    epochs=10, 
    batch_size=32, 
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_cnn, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predictions and Metrics
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=mapping.values(), yticklabels=mapping.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(Y_test, y_pred, target_names=mapping.values()))

# Save model weights
weights_file = 'cnn_model_weights.h5'
model.save_weights(weights_file)
print(f"Model weights saved to {weights_file}")
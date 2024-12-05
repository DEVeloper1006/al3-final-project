import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import resample
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Selected features
selected_features = [
    ' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
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
    'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 
    'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', 
    ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 
    'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
]

# Load the data
file_path = 'traffic_data.parquet'  # Replace with the actual path to your parquet file
data = pd.read_parquet(file_path)
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows where Flow Bytes or Flow Packets have NaN
data.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)

# Ensure no infinite or NaN values remain
assert not data.isin([np.inf, -np.inf]).any().any(), "Infinity values still exist!"
assert not data.isna().any().any(), "NaN values still exist!"

# Step 1: Encode labels
label_encoder = LabelEncoder()
data[' Label'] = label_encoder.fit_transform(data[' Label'])

# Step 2: Split the data
features = data[selected_features]
constant_features_idx = [28,40,44,45]
features = features.drop(features.columns[constant_features_idx], axis=1)
labels = data[' Label']

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Step 3: Downsampling and SMOTE for balancing the data
max_samples_per_class = 2500
X_train_resampled = []
y_train_resampled = []

for class_label in np.unique(y_train):
    X_class = X_train[y_train == class_label]
    y_class = y_train[y_train == class_label]
    if len(X_class) > max_samples_per_class:
        indices = np.random.choice(len(X_class), max_samples_per_class, replace=False)
        X_class = X_class.iloc[indices]
        y_class = y_class.iloc[indices]
    X_train_resampled.append(X_class)
    y_train_resampled.append(y_class)

X_train_resampled = pd.concat(X_train_resampled)
y_train_resampled = pd.concat(y_train_resampled)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_resampled, y_train_resampled)

print(f"Resampled dataset size: {Counter(y_train_balanced)}")

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 5: Feature Selection
k_best_features = 50  # Modify as needed
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
X_test_selected = selector.transform(X_test_scaled)

# Step 6: Ensemble SVM
print("Training Ensemble SVM...")
ensemble_classifiers = []
n_estimators = 5  # Number of SVM models in the ensemble

for i in range(n_estimators):
    # Bootstrap sampling
    print(f"Estimator: {i}")
    X_resampled, y_resampled = resample(X_train_selected, y_train_balanced, random_state=i)
    svm = SVC(kernel='linear', class_weight='balanced', C=0.5, gamma='scale', probability=True, random_state=i)
    svm.fit(X_resampled, y_resampled)
    ensemble_classifiers.append((f'svm_{i}', svm))

print("Now ensemble fitting")
# Use VotingClassifier to combine SVM predictions
ensemble = VotingClassifier(estimators=ensemble_classifiers, voting='soft', n_jobs=-1)
ensemble.fit(X_train_selected, y_train_balanced)

print("Now Predicting")
# Predictions
y_pred_ensemble = ensemble.predict(X_test_selected)

# Step 7: Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

f1 = f1_score(y_test, y_pred_ensemble, average='macro')
print(f"Macro F1 Score: {f1:.4f}")

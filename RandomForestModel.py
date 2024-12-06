# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from collections import Counter
from sklearn.svm import SVC  # Importing SVC for nonlinear SVM
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.model_selection import StratifiedKFold


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


# Load and preprocess data
file_path = 'traffic_data.parquet'
data = pd.read_parquet(file_path)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)

# Encode labels
label_encoder = LabelEncoder()
data[' Label'] = label_encoder.fit_transform(data[' Label'])

# Prepare features and labels
features = data[selected_features]
constant_features_idx = [28, 40, 44, 45]
features = features.drop(features.columns[constant_features_idx], axis=1)
labels = data[' Label']

# Initialize cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(features, labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    # Split data for current fold
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    # Balance classes
    benign_class = 0
    X_train_benign = X_train[y_train == benign_class]
    y_train_benign = y_train[y_train == benign_class]
    X_train_minority = X_train[y_train != benign_class]
    y_train_minority = y_train[y_train != benign_class]
    
    X_train_combined = np.vstack([X_train_benign, X_train_minority])
    y_train_combined = np.hstack([y_train_benign, y_train_minority])
    
    # Downsample to max_samples_per_class
    max_samples_per_class = 150000
    X_train_resampled = []
    y_train_resampled = []
    
    for class_label in np.unique(y_train_combined):
        mask = y_train_combined == class_label
        X_class = X_train_combined[mask]
        y_class = y_train_combined[mask]
        
        if len(X_class) > max_samples_per_class:
            indices = np.random.choice(len(X_class), max_samples_per_class, replace=False)
            X_class = X_class[indices]
            y_class = y_class[indices]
        
        X_train_resampled.append(X_class)
        y_train_resampled.append(y_class)
    
    X_train_resampled = np.vstack(X_train_resampled)
    y_train_resampled = np.hstack(y_train_resampled)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_resampled, y_train_resampled)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Select features
    selector = SelectKBest(score_func=f_classif, k=62)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Train and evaluate model
    model = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=42)
    model.fit(X_train_selected, y_train_balanced)
    
    y_pred = model.predict(X_test_selected)
    fold_f1 = f1_score(y_test, y_pred, average='macro')
    cv_scores.append(fold_f1)
    
    print(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Print final results
print("\nCross-validation results:")
print(f"Mean F1 Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
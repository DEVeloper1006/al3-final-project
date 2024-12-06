# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import ADASYN
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

# Load the data from the parquet file
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

# Step 3: Downsampling and ADASYN for balancing the data
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

adasyn = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_resampled, y_train_resampled)

print(f"Resampled dataset size: {Counter(y_train_balanced)}")

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 5: Feature Selection
k_best_features = 62  # Select the top 30 features (modify as needed)
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
X_test_selected = selector.transform(X_test_scaled)

# Step 6: One-vs-All (OvA) SVM Training
print("Training One-vs-All SVM...")
classifiers = {}
classes = y_train_balanced.unique()

for cls in classes:
    print(f"Training SVM for class {cls}")
    y_binary = (y_train_balanced == cls).astype(int)
    svc = SVC(random_state=42, class_weight='balanced', kernel='rbf', gamma='scale', C=1, probability=True)
    svc.fit(X_train_selected, y_binary)
    classifiers[cls] = svc

print("Training completed!")

# Step 7: Prediction
print("Evaluating the model...")
y_pred = []
for i in range(len(X_test_selected)):
    sample = X_test_selected[i].reshape(1, -1)
    confidences = {cls: clf.decision_function(sample)[0] for cls, clf in classifiers.items()}
    predicted_class = max(confidences, key=confidences.get)
    y_pred.append(predicted_class)

y_pred = np.array(y_pred)

# Step 8: Evaluation
print("Classification Report:")
test_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(test_report)

print("Confusion Matrix:")
test_conf_matrix = confusion_matrix(y_test, y_pred)
print(test_conf_matrix)

f1 = f1_score(y_test, y_pred, average='macro')
print(f"Macro F1 Score: {f1:.4f}")

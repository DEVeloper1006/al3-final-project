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
constant_features_idx = [28, 44]
# features = features.drop(features.columns[constant_features_idx], axis=1)
labels = data[' Label']

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Step 3: SMOTE resampling before feature selection
benign_class = 0 
minority_classes = y_train[y_train != benign_class].unique()

X_train_benign = X_train[y_train == benign_class]
y_train_benign = y_train[y_train == benign_class]
X_train_minority = X_train[y_train != benign_class]
y_train_minority = y_train[y_train != benign_class]


# Combine benign and minority class
X_train_combined = np.vstack([X_train_benign, X_train_minority])
y_train_combined = np.hstack([y_train_benign, y_train_minority])

X_train_balanced, y_train_balanced = X_train_combined, y_train_combined

# Convert y_train_balanced to Pandas Series
y_train_balanced = pd.Series(y_train_balanced)

# Step 1: Reduce the dataset size to 10,000 samples per class
class_counts = y_train_balanced.value_counts()
max_samples_per_class = 150000

# Initialize lists for the downsampled data
X_train_resampled = []
y_train_resampled = []

# Loop through each class to ensure 10,000 samples per class
for class_label in class_counts.index:
    # Extract the samples for the current class
    X_class = X_train_balanced[y_train_balanced == class_label]
    y_class = y_train_balanced[y_train_balanced == class_label]
    
    # Downsample the class if it has more than 10,000 samples
    if len(X_class) > max_samples_per_class:
        # Randomly select max_samples_per_class samples from X_class and y_class
        indices = np.random.choice(len(X_class), max_samples_per_class, replace=False)
        X_class = X_class[indices]
        y_class = y_class.iloc[indices]  # Use .iloc for indexing a Pandas Series
    
    # Append the (possibly downsampled) data to the lists
    X_train_resampled.append(X_class)
    y_train_resampled.append(y_class)

# Combine the resampled data for all classes
X_train_resampled = np.vstack(X_train_resampled)
y_train_resampled = np.hstack(y_train_resampled)

# Now apply SMOTE only to balance the entire dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_resampled, y_train_resampled)

print(f"Resampled dataset size: {Counter(y_train_balanced)}")

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

best_k = 0
best_f1_score = 0
#To test multiple feature sets, currently only loops for 62 cause I dont want t0 test that
for i in range(62, 63):
    # Step 5: Feature Selection (select top i features based on ANOVA F-statistic)
    selector = SelectKBest(score_func=f_classif, k=i)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features_mask = selector.get_support()  # Boolean mask of selected features
    selected_feature_names = features.columns[selected_features_mask]  # Get the selected feature names
    print(f"Selected Features ({i}):")
    print(selected_feature_names.tolist())


    # Using GridSearchCV to find the best combination of hyperparameters
    model = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)

  
    print("Fitting")
    # Training the model with the best parameters
    model.fit(X_train_scaled, y_train_balanced)
    print("Fitted")
    # Step 7: Evaluate the model
    print("Evaluating the model...")
    y_test_pred = model.predict(X_test_scaled)

    # Classification report
    test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(test_report)

    # Confusion matrix
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(test_conf_matrix)

    # F1 Score
    current_f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f"F1 Score: {current_f1:.4f}")

    if current_f1 > best_f1_score:
        best_f1_score = current_f1
        best_k = i
    
print(f"Best number of features: {best_k} with F1 score: {best_f1_score:.4f}")

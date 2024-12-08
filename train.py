import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier


def train_SVM():
    # Define attack types for each level
    level_1_labels = {
        'BENIGN': 0,
        'ATTACK': 1
    }


    group_labels = {
        'DoS/DDoS': ['DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'DDoS'],
        'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Bot'],
        'Reconnaissance': ['PortScan', 'Infiltration']
    }

    # Map the level 2 groups to numerical encoding
    group_mapping = {
        'DoS/DDoS': 0,
        'Brute Force': 1,
        'Reconnaissance': 2
    }
    

    # Function to create hierarchical labels
    def create_hierarchical_labels(data):
        """Create hierarchical labels for each level of classification"""
        
        # Level 1: Binary classification (BENIGN vs ATTACK)
        data['Level_1_Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Level 2: Attack group classification
        def get_group_label(attack_type):
            if attack_type == 'BENIGN':
                return -1
            for group, attacks in group_labels.items():
                if attack_type in attacks:
                    return group_mapping[group]
            return -1
        
        data['Level_2_Label'] = data[' Label'].apply(get_group_label)
        
        
        # Level 3: Specific attack type within group
        attack_type_mapping = {}
        for group, attacks in group_labels.items():
            for i, attack in enumerate(attacks):
                attack_type_mapping[attack] = i
        
        def get_specific_attack_label(row):
            if row[' Label'] == 'BENIGN':
                return -1
            return attack_type_mapping.get(row[' Label'], -1)
        
        data['Level_3_Label'] = data.apply(get_specific_attack_label, axis=1)
        
        return data

    def prepare_data_for_cv():
        print("Loading and preprocessing data...")
        file_path = 'train_features.parquet'
        data = pd.read_parquet(file_path)
        data = create_hierarchical_labels(data)

        # Balance the dataset
        balanced_samples = []
        max_samples_per_class = 5000
        for label in np.unique(data[' Label']):
            class_data = data[data[' Label'] == label]
            if label != 'BENIGN':
                if len(class_data) > max_samples_per_class:
                    class_data = class_data.sample(max_samples_per_class, random_state=42)
            else:
                if len(class_data) > max_samples_per_class:
                    class_data = class_data.sample(50000, random_state=42)
            balanced_samples.append(class_data)
        
        balanced_data = pd.concat(balanced_samples)
        
        X = balanced_data.drop(columns=[' Label', 'Level_1_Label', 'Level_2_Label', 'Level_3_Label'])
        y_level_1 = balanced_data['Level_1_Label']
        y_level_2 = balanced_data['Level_2_Label']
        y_level_3 = balanced_data['Level_3_Label']
        
        return X, y_level_1, y_level_2, y_level_3

    def cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3, n_splits=5):
        level_2_names = {
        0: 'DoS/DDoS',
        1: 'Brute Force',
        2: 'Reconnaissance'
        }
    
        level_3_names = {
            0: {  # DoS/DDoS attacks
                0: 'DoS GoldenEye',
                1: 'DoS Hulk',
                2: 'DoS Slowhttptest',
                3: 'DoS slowloris',
                4: 'DDoS'
            },
            1: {  # Brute Force attacks
                0: 'FTP-Patator',
                1: 'SSH-Patator',
                2: 'Heartbleed',
                3: 'Bot'
            },
            2: {  # Reconnaissance attacks
                0: 'PortScan',
                1: 'Infiltration'
            }
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        metrics = {
            'level_1': [],
            'level_2': [],
            'level_3': {0: [], 1: [], 2: []}
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_level_1)):
            
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_l1, y_test_l1 = y_level_1.iloc[train_idx], y_level_1.iloc[test_idx]
            y_train_l2, y_test_l2 = y_level_2.iloc[train_idx], y_level_2.iloc[test_idx]
            y_train_l3, y_test_l3 = y_level_3.iloc[train_idx], y_level_3.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Level 1: Binary classification
            level_1_model = SVC(kernel='rbf', random_state=42, C=1)
            level_1_model.fit(X_train_scaled, y_train_l1)
            l1_preds = level_1_model.predict(X_test_scaled)
            l1_report = classification_report(y_test_l1, l1_preds, target_names=['BENIGN', 'ATTACK'], output_dict=True)
            metrics['level_1'].append(l1_report)
            
            # Level 2: Attack group classification
            attack_mask_train = y_train_l1 == 1
            attack_mask_test = y_test_l1 == 1
            
            level_2_model = SVC(kernel='linear', random_state=42, C=0.5)
            valid_l2_mask_train = attack_mask_train & (y_train_l2 != -1)
            valid_l2_mask_test = attack_mask_test & (y_test_l2 != -1)
            
            if np.any(valid_l2_mask_train):
                level_2_model.fit(X_train_scaled[valid_l2_mask_train], 
                                y_train_l2[valid_l2_mask_train])
                l2_preds = level_2_model.predict(X_test_scaled[valid_l2_mask_test])
                l2_report = classification_report(y_test_l2[valid_l2_mask_test], l2_preds, target_names=[level_2_names[i] for i in range(3)], output_dict=True)
                metrics['level_2'].append(l2_report)
            level_3_models = {}
            # Level 3: Specific attack classification
            for group_id in range(3):
                print(group_id)
                group_mask_train = (y_train_l2 == group_id) & attack_mask_train
                group_mask_test = (y_test_l2 == group_id) & attack_mask_test
                
                if np.any(group_mask_train):
                    X_group_train = X_train_scaled[group_mask_train & (y_train_l3 != -1)]
                    y_group_train = y_train_l3[group_mask_train & (y_train_l3 != -1)]
                    
                    # Apply SMOTE
                    smote = SMOTE(random_state=42)
                    X_group_train_smote, y_group_train_smote = smote.fit_resample(
                        X_group_train, y_group_train)
                    
                    group_model = SVC(kernel='linear', random_state=42, C=0.01)
                    group_model.fit(X_group_train_smote, y_group_train_smote)
                    level_3_models[group_id] = group_model
                    valid_l3_mask_test = group_mask_test & (y_test_l3 != -1)
                    if np.any(valid_l3_mask_test):
                        valid_labels = sorted(list(set(y_test_l3[valid_l3_mask_test])))
                        l3_preds = group_model.predict(X_test_scaled[valid_l3_mask_test])
                        l3_report = classification_report(y_test_l3[valid_l3_mask_test], 
                                                    l3_preds, target_names=[level_3_names[group_id][i] for i in valid_labels], output_dict=True, labels=valid_labels)
                        metrics['level_3'][group_id].append(l3_report)
        
        return scaler, metrics, level_1_model, level_2_model, level_3_models

    def print_cv_results(metrics):
        
        # Level 1 results
        l1_accuracy = np.mean([m['accuracy'] for m in metrics['level_1']])
        l1_std = np.std([m['accuracy'] for m in metrics['level_1']])
        print(f"\nLevel 1 CV Results:")
        print(f"Average Accuracy: {l1_accuracy:.3f} ± {l1_std:.3f}")
        
        # Level 2 results
        if metrics['level_2']:
            l2_accuracy = np.mean([m['accuracy'] for m in metrics['level_2']])
            l2_std = np.std([m['accuracy'] for m in metrics['level_2']])
            print(f"\nLevel 2 CV Results:")
            print(f"Average Accuracy: {l2_accuracy:.3f} ± {l2_std:.3f}")
        
        # Level 3 results
        print("\nLevel 3 CV Results by Group:")
        for group_id in range(3):
            if metrics['level_3'][group_id]:
                l3_accuracy = np.mean([m['accuracy'] for m in metrics['level_3'][group_id]])
                l3_std = np.std([m['accuracy'] for m in metrics['level_3'][group_id]])
                print(f"Group {group_id} Average Accuracy: {l3_accuracy:.3f} ± {l3_std:.3f}")

    
    X, y_level_1, y_level_2, y_level_3 = prepare_data_for_cv()
    scaler, metrics, level_1_model, level_2_model, level_3_models = cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3)
    print_cv_results(metrics)

    # Save models to files
    with open('level_1_model.pkl', 'wb') as f:
        pickle.dump(level_1_model, f)
    with open('level_2_model.pkl', 'wb') as f:
        pickle.dump(level_2_model, f)
    with open('level_3_models.pkl', 'wb') as f:
        pickle.dump(level_3_models, f)
    with open('SVM_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Models saved successfully.")

def train_Random_Forest():
    # Load and preprocess data
    file_path = 'train_features.parquet'
    data = pd.read_parquet(file_path)

    # Encode labels
    label_encoder = LabelEncoder()
    data[' Label'] = label_encoder.fit_transform(data[' Label'])

    # Prepare features and labels
    features = data.drop(columns=[' Label'])
    constant_features_idx = [28, 40, 44, 45]
    features = features.drop(features.columns[constant_features_idx], axis=1)
    labels = data[' Label']

    # Initialize cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    best_scaler = None
    best_model = None
    best_f1_score = -1  # Initialize with a value lower than the lowest possible F1 score

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
        
        
        
        # Train and evaluate model
        model = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=42)
        model.fit(X_train_scaled, y_train_balanced)
        
        y_pred = model.predict(X_test_scaled)
        fold_f1 = f1_score(y_test, y_pred, average='macro')
        cv_scores.append(fold_f1)
        
        print(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        if fold_f1 > best_f1_score:
            best_f1_score = fold_f1
            best_model = model
            best_scaler = scaler

    # Print final results
    print("\nCross-validation results:")
    print(f"Mean F1 Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    if best_model is not None:
        with open("random_forest.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("Best model saved as 'random_forest.pkl'")
        with open("random_forest_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("Best model saved as 'random_forest_scaler.pkl'")

def train_cnn ():

    def apply_sampling(X, Y):
        # Reshape for sampling (flatten the data temporarily)
        X_flattened = X.reshape(X.shape[0], -1)
        
        # Separate benign (normal) class - typically class 0
        benign_class = 0
        benign_mask = Y == benign_class
        X_benign = X_flattened[benign_mask]
        Y_benign = Y[benign_mask]
        
        # Separate minority classes
        X_minority = X_flattened[~benign_mask]
        Y_minority = Y[~benign_mask]
        
        # Combine benign and minority classes
        X_combined = np.vstack([X_benign, X_minority])
        Y_combined = np.hstack([Y_benign, Y_minority])
        
        # Downsample to max_samples_per_class
        max_samples_per_class = 150000
        X_resampled = []
        Y_resampled = []
        
        for class_label in np.unique(Y_combined):
            mask = Y_combined == class_label
            X_class = X_combined[mask]
            Y_class = Y_combined[mask]
            
            # Downsample if class exceeds max_samples
            if len(X_class) > max_samples_per_class:
                indices = np.random.choice(len(X_class), max_samples_per_class, replace=False)
                X_class = X_class[indices]
                Y_class = Y_class[indices]
            
            X_resampled.append(X_class)
            Y_resampled.append(Y_class)
        
        X_resampled = np.vstack(X_resampled)
        Y_resampled = np.hstack(Y_resampled)
        
        # Apply SMOTE for final balancing
        ros = SMOTE(random_state=42)
        X_final, Y_final = ros.fit_resample(X_resampled, Y_resampled)
        
        # Shuffle to randomize the data order
        X_final, Y_final = shuffle(X_final, Y_final, random_state=42)
        
        # Reshape back to CNN-compatible shape
        X_final = X_final.reshape(X_final.shape[0], X.shape[1], X.shape[2])
        
        return X_final, Y_final

    def build_cnn(input_shape, num_classes):
        model = Sequential([
            # First Convolutional Block
            Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.1), input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            # Second Convolutional Block
            Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.1)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            # Third Convolutional Block
            Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.1)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),

            # Flatten before passing to fully connected layers
            Flatten(),

            # Dense Layer
            Dense(512, activation='relu', kernel_regularizer=l2(0.1)),
            Dropout(0.5),

            # Output Layer for multi-class classification
            Dense(num_classes, activation='softmax')
            ])
        
        optimizer = Adam(learning_rate=0.001)
        # Compile the model
        model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model

    def map_labels (df, col_name):
        mapping = {}
        val = 0
        for label in df[col_name].unique():
            mapping[label] = val
            val += 1
        df[col_name] = df[col_name].map(mapping)
        return df, mapping
    
    training_data = pd.read_parquet("train_features.parquet")
    training_data, mapping = map_labels(training_data, ' Label')

    new_map = {}
    for key, value in mapping.items():
        new_map[value] = key
    
    X = training_data.iloc[:, :-1].to_numpy()
    Y = training_data.iloc[:, -1].to_numpy()

    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    # # Reshape for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)    
    
    # Apply advanced sampling
    X_train_resampled, Y_train_resampled = apply_sampling(X_train_cnn, Y_train)
    
    # Print class distribution
    print("Resampled Training Data Class Distribution:")
    unique, counts = np.unique(Y_train_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u} ('{new_map[u]}'): {c} samples")
    
    # Callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6
    )
    
    # Build and train the model
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
    print(input_shape)
    num_classes = len(np.unique(Y_train))

    
    model = build_cnn(input_shape, num_classes)
    
    history = model.fit(
        X_train_resampled, Y_train_resampled,
        validation_data=(X_val_cnn, Y_val),
        epochs=10,  # Increased epochs for more complex training
        batch_size=16,  # Adjusted batch size
        callbacks=[early_stopping, lr_scheduler]
    )
    
    weights = model.get_weights()
    with open("cnn_model_weights.pkl", 'wb') as f:
        pickle.dump(weights, f)
    print("Model weights saved to the file!")


print("TRAINING RANDOM FOREST")
train_Random_Forest()
print("TRAINING SVM")
train_SVM()
print("TRAINING CNN")
train_cnn()
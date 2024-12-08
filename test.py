import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam



def test_svm():
    def test_hierarchical_models(test_features_path, scaler_path, level1_model_path, level2_model_path, level3_models_path):
        """
        Perform hierarchical predictions on test data using pre-trained models.
        
        Args:
            test_features_path (str): Path to the test_features.parquet file
            scaler_path (str): Path to the saved StandardScaler
            level1_model_path (str): Path to the saved Level 1 model
            level2_model_path (str): Path to the saved Level 2 model
            level3_models_path (str): Path to the saved Level 3 models dictionary
        
        Returns:
            dict: Dictionary containing predictions and classification reports for each level
        """
        # Define mappings for interpretable results
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

        # Load test data
        print("Loading test data...")
        test_data = pd.read_parquet(test_features_path)
        
        # Create hierarchical labels
        print("Creating hierarchical labels...")
        group_labels = {
            'DoS/DDoS': ['DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'DDoS'],
            'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Bot'],
            'Reconnaissance': ['PortScan', 'Infiltration']
        }
        
        group_mapping = {
            'DoS/DDoS': 0,
            'Brute Force': 1,
            'Reconnaissance': 2
        }
        
        # Create Level 1 labels
        test_data['Level_1_Label'] = test_data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Create Level 2 labels
        def get_group_label(attack_type):
            if attack_type == 'BENIGN':
                return -1
            for group, attacks in group_labels.items():
                if attack_type in attacks:
                    return group_mapping[group]
            return -1
        
        test_data['Level_2_Label'] = test_data[' Label'].apply(get_group_label)
        
        # Create Level 3 labels
        attack_type_mapping = {}
        for group, attacks in group_labels.items():
            for i, attack in enumerate(attacks):
                attack_type_mapping[attack] = i
        
        def get_specific_attack_label(row):
            if row[' Label'] == 'BENIGN':
                return -1
            return attack_type_mapping.get(row[' Label'], -1)
        
        test_data['Level_3_Label'] = test_data.apply(get_specific_attack_label, axis=1)
        
        # Load models and scaler
        print("Loading models and scaler...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(level1_model_path, 'rb') as f:
            level_1_model = pickle.load(f)
        with open(level2_model_path, 'rb') as f:
            level_2_model = pickle.load(f)
        with open(level3_models_path, 'rb') as f:
            level_3_models = pickle.load(f)
        
        # Prepare features
        X_test = test_data.drop(columns=[' Label', 'Level_1_Label', 'Level_2_Label', 'Level_3_Label'])
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Level 1 predictions
        print("Making Level 1 predictions...")
        l1_preds = level_1_model.predict(X_test_scaled)
        results['level_1'] = {
            'predictions': l1_preds,
            'report': classification_report(test_data['Level_1_Label'], l1_preds, 
                                        target_names=['BENIGN', 'ATTACK'], 
                                        output_dict=True)
        }
        
        # Level 2 predictions (only for attacks)
        print("Making Level 2 predictions...")
        attack_mask = l1_preds == 1
        if np.any(attack_mask):
            l2_preds = level_2_model.predict(X_test_scaled[attack_mask])
            valid_l2_mask = test_data['Level_2_Label'][attack_mask] != -1
            if np.any(valid_l2_mask):
                results['level_2'] = {
                    'predictions': l2_preds,
                    'report': classification_report(
                        test_data['Level_2_Label'][attack_mask][valid_l2_mask],
                        l2_preds[valid_l2_mask],
                        target_names=[level_2_names[i] for i in range(3)],
                        output_dict=True
                    )
                }
        
        # Level 3 predictions (specific attacks within each group)
        print("Making Level 3 predictions...")
        results['level_3'] = {}
        for group_id in range(3):
            group_mask = (test_data['Level_2_Label'] == group_id) & attack_mask
            if np.any(group_mask):
                valid_l3_mask = test_data['Level_3_Label'][group_mask] != -1
                if np.any(valid_l3_mask):
                    group_model = level_3_models[group_id]
                    l3_preds = group_model.predict(X_test_scaled[group_mask][valid_l3_mask])
                    valid_labels = sorted(list(set(test_data['Level_3_Label'][group_mask][valid_l3_mask])))
                    results['level_3'][group_id] = {
                        'predictions': l3_preds,
                        'report': classification_report(
                            test_data['Level_3_Label'][group_mask][valid_l3_mask],
                            l3_preds,
                            target_names=[level_3_names[group_id][i] for i in valid_labels],
                            output_dict=True,
                            labels=valid_labels
                        )
                    }
        
        return results

    def print_test_results(results):
        """Print the classification reports for each level"""
        print("\nLevel 1 Results:")
        print(f"Accuracy: {results['level_1']['report']['accuracy']:.3f}")
        print("\nClassification Report:")
        for label in ['BENIGN', 'ATTACK']:
            metrics = results['level_1']['report'][label]
            print(f"\n{label}:")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-score: {metrics['f1-score']:.3f}")
        
        if 'level_2' in results:
            print("\nLevel 2 Results:")
            print(f"Accuracy: {results['level_2']['report']['accuracy']:.3f}")
            print("\nClassification Report:")
            for group in ['DoS/DDoS', 'Brute Force', 'Reconnaissance']:
                metrics = results['level_2']['report'][group]
                print(f"\n{group}:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1-score: {metrics['f1-score']:.3f}")
        
        if 'level_3' in results:
            print("\nLevel 3 Results:")
            for group_id, group_results in results['level_3'].items():
                print(f"\nGroup {group_id} Results:")
                print(f"Accuracy: {group_results['report']['accuracy']:.3f}")
                print("\nClassification Report:")
                for attack_type, metrics in group_results['report'].items():
                    if attack_type not in ['accuracy', 'macro avg', 'weighted avg']:
                        print(f"\n{attack_type}:")
                        print(f"Precision: {metrics['precision']:.3f}")
                        print(f"Recall: {metrics['recall']:.3f}")
                        print(f"F1-score: {metrics['f1-score']:.3f}")
    
    results = test_hierarchical_models(
    'test_features.parquet',
    'SVM_scaler.pkl',
    'level_1_model.pkl',
    'level_2_model.pkl',
    'level_3_models.pkl'
    )   
    print_test_results(results)

def test_random_forest(test_file_path, model_path, scaler_path):
    """
    Test a trained Random Forest model on new data.
    
    Args:
        test_file_path (str): Path to the test features parquet file
        model_path (str): Path to the saved model pickle file
        scaler_path (str): Path to the saved StandardScaler pickle file
    
    Returns:
        dict: Dictionary containing predictions and evaluation metrics
    """
    # Load test data
    print("Loading test data...")
    test_data = pd.read_parquet(test_file_path)
    
    # Encode labels if present (for evaluation)
    label_encoder = LabelEncoder()
    if ' Label' in test_data.columns:
        test_labels = label_encoder.fit_transform(test_data[' Label'])
        test_features = test_data.drop(columns=[' Label'])
    else:
        test_labels = None
        test_features = test_data
    
    # Remove constant features that were dropped during training
    constant_features_idx = [28, 40, 44, 45]
    test_features = test_features.drop(test_features.columns[constant_features_idx], axis=1)
    
    # Load the preprocessing objects and model
    print("Loading model and preprocessing objects...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    
    # Apply the same preprocessing as during training
    print("Preprocessing test data...")
    X_test_scaled = scaler.transform(test_features)

    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)
    
    # Prepare results
    results = {
        'predictions': predictions,
    }
    
    # Calculate metrics if labels are available
    if test_labels is not None:
        print("\nEvaluation Metrics:")
        results['f1_score'] = f1_score(test_labels, predictions, average='macro')
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        print("\nClassification Report:")
        class_names = label_encoder.classes_ if hasattr(label_encoder, 'classes_') else None
        report = classification_report(test_labels, predictions, 
                                    target_names=class_names)
        print(report)
        results['classification_report'] = report
    
    return results

def test_cnn():
    def map_labels (df, col_name):
        mapping = {}
        val = 0
        for label in df[col_name].unique():
            mapping[label] = val
            val += 1
        df[col_name] = df[col_name].map(mapping)
        return df, mapping
    
    # Load data
    testing_data = pd.read_parquet("test_features.parquet")
    testing_data, mapping = map_labels(testing_data, ' Label')
    
    new_map = {}
    for key, value in mapping.items():
        new_map[value] = key
    
    X_test = testing_data.iloc[:, :-1].to_numpy()
    Y_test = testing_data.iloc[:, -1].to_numpy()
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Load Model
    model = Sequential([
            # First Convolutional Block
            Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.1), input_shape=(67, 1)),
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
            Dropout(0.7),

            # Output Layer for multi-class classification
            Dense(12, activation='softmax')
            ])
    optimizer = Adam(learning_rate=0.001)
        # Compile the model
    model.compile(
        optimizer=optimizer,             
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    weights_file = "cnn_model_weights.pkl"
    with open(weights_file, "rb") as f:
        loaded_weights = pickle.load(f)
    
    model.set_weights(loaded_weights)
    test_loss, test_accuracy = model.evaluate(X_test_cnn, Y_test)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    # Predictions
    y_pred_probs = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=new_map.values(), yticklabels=new_map.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred, target_names=new_map.values()))


print("RANDOM FOREST TEST")
test_random_forest('test_features.parquet', 'random_forest.pkl', 'random_forest_scaler.pkl')
print("SVM TEST")
test_svm()
print("CNN TEST")
test_cnn()
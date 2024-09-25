import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import xgboost as xgb


# Define the number of frames per sequence
sequence_length = 250

# Function to prepare sequences and split into 5 equal parts
def prepare_sequences_and_split(data_list, sequence_length, n_splits=5, step_size=1):
    split_sequences = [[] for _ in range(n_splits)]
    for data in data_list:
        split_size = len(data) // n_splits
        split_points = [i * split_size for i in range(n_splits)] + [len(data)]

        for split_idx in range(n_splits):
            split_data = data.iloc[split_points[split_idx]:split_points[split_idx + 1]]
            split_seq = []
            for i in range(0, len(split_data) - sequence_length + 1, step_size):
                sequence = split_data.iloc[i:i + sequence_length]
                split_seq.append(sequence.values)
            split_sequences[split_idx].append(split_seq)

    split_sequences = [np.vstack(split_seq) for split_seq in split_sequences]
    return split_sequences

# Load data from Excel files
# Add paths to your Excel files dataset
infected_files = ['your_infected_dataset_paths_here']
noninfected_files = ['your_noninfected_dataset_paths_here']

infected_data_list = [pd.read_excel(file, engine='openpyxl') for file in infected_files]
noninfected_data_list = [pd.read_excel(file, engine='openpyxl') for file in noninfected_files]

# Prepare sequences and split them
splits_infected = prepare_sequences_and_split(infected_data_list, sequence_length)
splits_noninfected = prepare_sequences_and_split(noninfected_data_list, sequence_length)

# Combine splits into k-fold datasets
k = 5
folds = []

for i in range(k):
    # Create validation data
    X_val_infected = splits_infected[i]
    X_val_noninfected = splits_noninfected[i]
    
    # Balance validation data
    if X_val_infected.shape[0] > X_val_noninfected.shape[0]:
        indices = np.random.choice(X_val_infected.shape[0], X_val_noninfected.shape[0], replace=False)
        X_val_infected = X_val_infected[indices]
    else:
        indices = np.random.choice(X_val_noninfected.shape[0], X_val_infected.shape[0], replace=False)
        X_val_noninfected = X_val_noninfected[indices]

    X_val = np.vstack([X_val_infected, X_val_noninfected])
    y_val = np.hstack([np.ones(X_val_infected.shape[0]), np.zeros(X_val_noninfected.shape[0])])
    
    # Create training data by excluding the current fold
    X_train_infected = np.vstack([splits_infected[j] for j in range(k) if j != i])
    X_train_noninfected = np.vstack([splits_noninfected[j] for j in range(k) if j != i])
    
    # Balance training data
    if X_train_infected.shape[0] > X_train_noninfected.shape[0]:
        indices = np.random.choice(X_train_infected.shape[0], X_train_noninfected.shape[0], replace=False)
        X_train_infected = X_train_infected[indices]
    else:
        indices = np.random.choice(X_train_noninfected.shape[0], X_train_infected.shape[0], replace=False)
        X_train_noninfected = X_train_noninfected[indices]
    
    X_train = np.vstack([X_train_infected, X_train_noninfected])
    y_train = np.hstack([np.ones(X_train_infected.shape[0]), np.zeros(X_train_noninfected.shape[0])])
    
    # Shuffle training data
    train_indices = np.arange(X_train.shape[0])
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    # Reshape for CNN input
    X_train = X_train.reshape((X_train.shape[0], sequence_length, 3))
    X_val = X_val.reshape((X_val.shape[0], sequence_length, 3))
    
    folds.append((X_train, y_train, X_val, y_val))

# Define and build the optimized CNN model with 3 convolutional layers and 3 dense layers
def build_regularized_model():
    inputs = Input(shape=(sequence_length, 3))
    x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, strides=2, activation='relu')(x)  # 3 convolutional layers
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)  # Optimized dense layer with 128 neurons
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)  # 3 dense layers
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # Optimized learning rate
    return model

# Define other classifiers with optimized hyperparameters
classifiers = {
    "XGBoost": xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.001, max_depth=7, n_estimators=100, subsample=0.7, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.01, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=5, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_split=5, random_state=42),
    "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    "Logistic Regression": LogisticRegression(C=1, max_iter=10000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, learning_rate_init=0.001, max_iter=10000, random_state=42)
}

# Store results
cnn_results = []
classifier_results = {name: [] for name in classifiers}
classifier_results["CNN + XGBoost"] = []

# Train and evaluate models on each fold
for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(folds):
    print(f"\nResults for Fold {fold_idx + 1}:")

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)
    
    # Train optimized CNN model
    regularized_cnn_model = build_regularized_model()
    regularized_cnn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=512, verbose=0)  # Optimized epochs and batch size
    
    # Evaluate CNN model
    cnn_y_val_pred = regularized_cnn_model.predict(X_val_scaled).flatten()
    cnn_y_val_pred_class = (cnn_y_val_pred > 0.5).astype(int)
    
    cnn_results.append({
        "accuracy": accuracy_score(y_val, cnn_y_val_pred_class),
        "precision": precision_score(y_val, cnn_y_val_pred_class),
        "recall": recall_score(y_val, cnn_y_val_pred_class),
        "f1": f1_score(y_val, cnn_y_val_pred_class),
        "confusion_matrix": confusion_matrix(y_val, cnn_y_val_pred_class)
    })
    
    print(f"  CNN Model: Accuracy={accuracy_score(y_val, cnn_y_val_pred_class):.4f}, Precision={precision_score(y_val, cnn_y_val_pred_class):.4f}, Recall={recall_score(y_val, cnn_y_val_pred_class):.4f}, F1 Score={f1_score(y_val, cnn_y_val_pred_class):.4f}")

    # Train and evaluate other classifiers
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)
        y_val_pred = clf.predict_proba(X_val_scaled.reshape(X_val_scaled.shape[0], -1))[:, 1]
        y_val_pred_class = (y_val_pred > 0.5).astype(int)
        
        classifier_results[name].append({
            "accuracy": accuracy_score(y_val, y_val_pred_class),
            "precision": precision_score(y_val, y_val_pred_class),
            "recall": recall_score(y_val, y_val_pred_class),
            "f1": f1_score(y_val, y_val_pred_class),
            "confusion_matrix": confusion_matrix(y_val, y_val_pred_class)
        })
        
        print(f"  {name} Model: Accuracy={accuracy_score(y_val, y_val_pred_class):.4f}, Precision={precision_score(y_val, y_val_pred_class):.4f}, Recall={recall_score(y_val, y_val_pred_class):.4f}, F1 Score={f1_score(y_val, y_val_pred_class):.4f}")
    
    # CNN + XGBoost combined model
    cnn_features_train = regularized_cnn_model.predict(X_train_scaled).flatten().reshape(-1, 1)
    cnn_features_val = regularized_cnn_model.predict(X_val_scaled).flatten().reshape(-1, 1)
    
    xgb_combined = xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.001, max_depth=7, n_estimators=100, subsample=0.7, random_state=42)
    xgb_combined.fit(cnn_features_train, y_train)
    y_val_combined_pred = xgb_combined.predict_proba(cnn_features_val)[:, 1]
    y_val_combined_pred_class = (y_val_combined_pred > 0.5).astype(int)
    
    classifier_results["CNN + XGBoost"].append({
        "accuracy": accuracy_score(y_val, y_val_combined_pred_class),
        "precision": precision_score(y_val, y_val_combined_pred_class),
        "recall": recall_score(y_val, y_val_combined_pred_class),
        "f1": f1_score(y_val, y_val_combined_pred_class),
        "confusion_matrix": confusion_matrix(y_val, y_val_combined_pred_class)
    })
    
    print(f"  CNN + XGBoost Model: Accuracy={accuracy_score(y_val, y_val_combined_pred_class):.4f}, Precision={precision_score(y_val, y_val_combined_pred_class):.4f}, Recall={recall_score(y_val, y_val_combined_pred_class):.4f}, F1 Score={f1_score(y_val, y_val_combined_pred_class):.4f}")

# Calculate mean results for classifiers
mean_results = {"Model": [], "Fold": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

for name in classifiers:
    for fold_idx, result in enumerate(classifier_results[name]):
        mean_results["Model"].append(name)
        mean_results["Fold"].append(fold_idx + 1)
        mean_results["Accuracy"].append(result["accuracy"])
        mean_results["Precision"].append(result["precision"])
        mean_results["Recall"].append(result["recall"])
        mean_results["F1 Score"].append(result["f1"])

    mean_accuracy = np.mean([res["accuracy"] for res in classifier_results[name]])
    mean_precision = np.mean([res["precision"] for res in classifier_results[name]])
    mean_recall = np.mean([res["recall"] for res in classifier_results[name]])
    mean_f1 = np.mean([res["f1"] for res in classifier_results[name]])
    
    mean_results["Model"].append(name + " (Mean)")
    mean_results["Fold"].append("Mean")
    mean_results["Accuracy"].append(mean_accuracy)
    mean_results["Precision"].append(mean_precision)
    mean_results["Recall"].append(mean_recall)
    mean_results["F1 Score"].append(mean_f1)
    
    print(f"{name} Model: Mean Accuracy={mean_accuracy:.4f}, Mean Precision={mean_precision:.4f}, Mean Recall={mean_recall:.4f}, Mean F1 Score={mean_f1:.4f}")

# Calculate mean results for CNN+XGBoost
cnn_xgb_mean_results = {
    "accuracy": np.mean([res["accuracy"] for res in classifier_results["CNN + XGBoost"]]),
    "precision": np.mean([res["precision"] for res in classifier_results["CNN + XGBoost"]]),
    "recall": np.mean([res["recall"] for res in classifier_results["CNN + XGBoost"]]),
    "f1": np.mean([res["f1"] for res in classifier_results["CNN + XGBoost"]])
}

mean_results["Model"].extend(["CNN + XGBoost"] * k + ["CNN + XGBoost (Mean)"])
mean_results["Fold"].extend(list(range(1, k + 1)) + ["Mean"])
mean_results["Accuracy"].extend([res["accuracy"] for res in classifier_results["CNN + XGBoost"]] + [cnn_xgb_mean_results["accuracy"]])
mean_results["Precision"].extend([res["precision"] for res in classifier_results["CNN + XGBoost"]] + [cnn_xgb_mean_results["precision"]])
mean_results["Recall"].extend([res["recall"] for res in classifier_results["CNN + XGBoost"]] + [cnn_xgb_mean_results["recall"]])
mean_results["F1 Score"].extend([res["f1"] for res in classifier_results["CNN + XGBoost"]] + [cnn_xgb_mean_results["f1"]])

print(f"CNN + XGBoost Model: Mean Accuracy={cnn_xgb_mean_results['accuracy']:.4f}, Mean Precision={cnn_xgb_mean_results['precision']:.4f}, Mean Recall={cnn_xgb_mean_results['recall']:.4f}, Mean F1 Score={cnn_xgb_mean_results['f1']:.4f}")

# Save results to a CSV file
mean_results_df = pd.DataFrame(mean_results)
mean_results_df.to_csv("model_performance_results.csv", index=False)

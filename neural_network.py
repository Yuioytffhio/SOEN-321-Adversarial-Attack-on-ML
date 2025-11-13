import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os # Import the os module to check for file existence

# --- 1. Define Column Names and Constants ---
# Based on the official NSL-KDD dataset documentation
# These are the 41 features + 1 label column
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Identify which columns are categorical (text) and which are numeric
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
# All other columns (except 'label' and 'difficulty') are numeric
NUMERIC_COLS = list(set(COLUMNS[:-2]) - set(CATEGORICAL_COLS))

# File path for saving the trained Keras model
MODEL_FILE = 'kdd_detection_model.keras'

# --- 2. Load Data ---
def load_data(train_path, test_path):
    """Loads the NSL-KDD train and test datasets."""
    try:
        # Load data, skipping the last 'difficulty' column (index 42)
        train_df = pd.read_csv(train_path, 
                               names=COLUMNS[:-1],  # Use the first 42 names (features + label)
                               usecols=range(42))   # Read the first 42 columns by index (0-41)
        
        test_df = pd.read_csv(test_path, 
                              names=COLUMNS[:-1],  # Use the first 42 names
                              usecols=range(42))   # Read the first 42 columns by index
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download 'KDDTrain+.txt' and 'KDDTest+.txt' from:")
        print("https://www.kaggle.com/datasets/hassan06/nslkdd?resource=download")
        print("and place them in the same directory as this script.")
        sys.exit(1)
        
    return train_df, test_df


# --- 3. Preprocess Data ---
def preprocess(train_df, test_df):
    """Applies One-Hot Encoding and Standard Scaling."""
    
    print("Starting preprocessing...")
    
    # --- 3.1. Handle Labels (Target Variable) ---
    # Simplify to binary classification: 0 = 'normal', 1 = 'attack'
    y_train = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
    y_test = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
    
    # Drop the label column from the feature dataframes
    X_train = train_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)

    # --- 3.2. One-Hot Encoding for Categorical Features ---
    # Combine train and test for consistent one-hot encoding
    combined_df = pd.concat([X_train, X_test], axis=0)
    
    # The get_dummies function creates new columns for each unique value
    # in the categorical columns.
    combined_df_encoded = pd.get_dummies(combined_df, columns=CATEGORICAL_COLS, dtype=float)
    
    # Separate back into train and test
    X_train_encoded = combined_df_encoded.iloc[:len(X_train)]
    X_test_encoded = combined_df_encoded.iloc[len(X_train):]
    
    print(f"Features transformed by one-hot encoding. New feature count: {X_train_encoded.shape[1]}")

    # --- 3.3. Standard Scaling for Numeric Features ---
    # We must scale *after* one-hot encoding
    # Identify the new set of numeric columns (original ones)
    # The one-hot encoded columns are already on a 0/1 scale
    
    scaler = StandardScaler()
    
    # Fit *only* on the training data's numeric columns
    scaler.fit(X_train_encoded[NUMERIC_COLS])
    
    # Transform both train and test data
    X_train_encoded[NUMERIC_COLS] = scaler.transform(X_train_encoded[NUMERIC_COLS])
    X_test_encoded[NUMERIC_COLS] = scaler.transform(X_test_encoded[NUMERIC_COLS])
    
    print("Numeric features scaled.")
    
    return X_train_encoded.values, X_test_encoded.values, y_train, y_test


# --- 4. Build the Neural Network ---
def build_model(input_shape):
    """Creates a Keras Sequential model."""
    
    model = keras.Sequential([
        # Input layer: Must match the number of features
        layers.Input(shape=(input_shape,)),
        
        # Hidden layer 1
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout for regularization
        
        # Hidden layer 2
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer: 1 neuron with sigmoid activation for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# --- 5. Main Execution ---
def main():
    # Define file paths
    TRAIN_FILE = "data/KDDTrain+.txt"
    TEST_FILE = "data/KDDTest+.txt"
    
    # Load and preprocess the data
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    X_train, X_test, y_train, y_test = preprocess(train_df, test_df)
    
    # Get the input shape for the model
    input_shape = X_train.shape[1]
    
    # --- Persistence Logic: Load or Train ---
    if os.path.exists(MODEL_FILE):
        # 5.1. Load existing model
        print(f"\n--- Loading Model from '{MODEL_FILE}' ---")
        try:
            model = keras.models.load_model(MODEL_FILE)
            print("Model loaded successfully. Skipping training.")
        except Exception as e:
            print(f"Error loading model: {e}. Rebuilding and training model.")
            model = build_model(input_shape)
            
            # Train the model
            print("\n--- Starting Model Training ---")
            history = model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=64,
                validation_split=0.1,
                verbose=1
            )
            # Save the newly trained model
            model.save(MODEL_FILE)
            print(f"\nModel trained and saved to '{MODEL_FILE}'.")
    else:
        # 5.2. Build, Train, and Save
        print(f"\n--- '{MODEL_FILE}' not found. Building and Training Model ---")
        model = build_model(input_shape)
        model.summary()
        
        # Train the model
        print("\n--- Starting Model Training ---")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.1, # Use 10% of training data for validation
            verbose=1
        )
        
        # Save the trained model
        model.save(MODEL_FILE)
        print(f"\nModel trained and saved to '{MODEL_FILE}'.")
    
    # Evaluate the model on the unseen test data (runs every time)
    print("\n--- Evaluating Model on Test Data ---")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    
    # Makes predictions on all values in KDDTest+.txt
    print("\n--- Making Predictions ---")

    correct_predictions = 0
    num_of_predictions = 22544
    predictions = model.predict(X_test[:num_of_predictions])
    
    for i in range(num_of_predictions):
        pred_class = "Attack" if predictions[i] > 0.5 else "Normal"
        actual_class = "Attack" if y_test[i] == 1 else "Normal"
        print(f"Sample {i+1}: Predicted={pred_class} (Raw: {predictions[i][0]:.3f}) | Actual={actual_class}")

        if pred_class == actual_class:
            correct_predictions += 1

    print(f"\nModel accuracy: {(correct_predictions/num_of_predictions)*100:.2f}%")


if __name__ == "__main__":
    main()
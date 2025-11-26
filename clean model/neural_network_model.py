import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------
# LOAD BALANCED DATA
# -----------------------------------
df = pd.read_csv("data/KDDTrain_balanced.txt", header=None)

label_col = 41   # attack vs normal

# -----------------------------------
# SEPARATE FEATURES & LABEL
# -----------------------------------
X = df.drop(columns=[label_col])
y = df[label_col].apply(lambda x: 0 if x == "normal" else 1)   # binary labels

# -----------------------------------
# IDENTIFY CATEGORICAL & NUMERIC COLS
# -----------------------------------
categorical_cols = [1, 2, 3, 6, 11, 20, 21]
numeric_cols = sorted([i for i in range(X.shape[1]) if i not in categorical_cols])



# -----------------------------------
# PREPROCESSING PIPELINE
# OneHotEncode categorical + scale numeric
# -----------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numerical", StandardScaler(), numeric_cols)
    ]
)

# Fit-transform all X
X_processed = preprocessor.fit_transform(X)

# Convert to dense if sparse
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# -----------------------------------
# TRAIN/TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# -----------------------------------
# BUILD THE NEURAL NETWORK
# -----------------------------------
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')   # binary classification
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=1024,
    validation_split=0.2,
    verbose=1
)

# -----------------------------------
# EVALUATE MODEL
# -----------------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# -----------------------------------
# SAVE MODEL
# -----------------------------------
model.save("clean model/neural_model.keras")
print("\nModel saved as neural_model.keras")

# -----------------------------------
# MAKE PREDICTIONS
# -----------------------------------
predictions = model.predict(X_test[:20])
print("\nExample predictions (0=normal, 1=attack):")
print(predictions)



# Make predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)  # convert probabilities to 0/1

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Detailed metrics: precision, recall, F1-score
report = classification_report(y_test, y_pred, target_names=["normal", "attack"])
print("\nClassification Report:")
print(report)
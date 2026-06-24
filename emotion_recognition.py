import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

# ==============================================================================
# PHASE 1: FEATURE EXTRACTION
# ==============================================================================
def extract_mfcc(file_path, n_mfcc=40):
    """
    Loads an audio file and extracts its mean MFCC features.
    """
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Take the mean across time (axis=1) to get a fixed-size vector
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ==============================================================================
# PHASE 2: DATA LOADING AND PREPROCESSING
# ==============================================================================
# RAVDESS filename emotion mapping
emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_dataset(data_dir):
    """
    Loops through the RAVDESS directory, extracts features, and collects labels.
    """
    X, y = [], []
    print("Starting feature extraction... This might take a few minutes.")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Parse emotion identifier from RAVDESS filename (e.g., 03-01-01-...)
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_mapping:
                        emotion = emotion_mapping[emotion_code]
                        
                        features = extract_mfcc(file_path)
                        if features is not None:
                            X.append(features)
                            y.append(emotion)
                            
    print(f"Successfully processed {len(X)} audio files.")
    return np.array(X), np.array(y)

# ==============================================================================
# PHASE 3: MODEL ARCHITECTURE
# ==============================================================================
def build_cnn_model(input_shape, num_classes):
    """
    Creates a 1D Convolutional Neural Network for MFCC feature classification.
    """
    model = Sequential([
        # First Conv Layer
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second Conv Layer
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # Points directly to the current folder containing the Actor folders
    DATA_DIR = "." 
    
    # 1. Load and process data
    X, y = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        print("[ERROR] No audio files found. Double check your folder placement!")
    else:
        # 2. Encode labels targets to one-hot vectors
        label_encoder = LabelEncoder()
        y_encoded = to_categorical(label_encoder.fit_transform(y))
        num_classes = len(label_encoder.classes_)
        
        # 3. Train/Test Split (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # 4. Reshape data for 1D CNN input layer -> (batch, features, 1 channel)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        
        # 5. Build model
        input_shape = (X_train.shape[1], 1)
        model = build_cnn_model(input_shape, num_classes)
        model.summary()
        
        # 6. Train model
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_data=(X_test, y_test)
        )
        
        # 7. Evaluate performance
        print("\nEvaluating model performance...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Final Test Loss: {loss:.4f}")
        print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

        # 8. Save the model for your live presentation demo
        model.save("speech_emotion_model.h5")
        print("Model saved successfully as speech_emotion_model.h5!")
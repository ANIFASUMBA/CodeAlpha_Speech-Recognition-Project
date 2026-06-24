import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

# The exact 8 emotions our model learned to recognize
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_mfcc(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return None

if __name__ == "__main__":
    # 1. Load the pre-trained model instantly
    if not os.path.exists("speech_emotion_model.h5"):
        print("[ERROR] speech_emotion_model.h5 not found! Please save your model first.")
        exit()
        
    model = load_model("speech_emotion_model.h5")
    
    # 2. Specify the file you want to test
    # Put a test audio file in your folder and type its name here!
    TEST_FILE = "test_voice.wav" 
    
    if not os.path.exists(TEST_FILE):
        print(f"[ERROR] Please place an audio file named '{TEST_FILE}' in this folder to test.")
    else:
        print(f"\nAnalyzing vocal features of '{TEST_FILE}'...")
        features = extract_mfcc(TEST_FILE)
        
        if features is not None:
            # Reshape features to match the 1D CNN input format
            features = np.expand_dims(features, axis=(0, -1))
            
            # Make prediction
            predictions = model.predict(features, verbose=0)
            predicted_class_index = np.argmax(predictions)
            predicted_emotion = EMOTIONS[predicted_class_index]
            confidence = predictions[0][predicted_class_index] * 100
            
            # Output the presentation result
            print("\n" + "="*40)
            print(f"  PREDICTED EMOTION : {predicted_emotion.upper()}")
            print(f"  CONFIDENCE SCORE  : {confidence:.2f}%")
            print("="*40)
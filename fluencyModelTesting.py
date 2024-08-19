import librosa
import numpy as np
import soundfile as sf
import joblib

# Function to extract features from a single audio file
def extract_features(file_path):
    try:
        # Use soundfile to read the audio file
        X, sample_rate = sf.read(file_path)

        if X.ndim > 1:
            X = X[:, 0]

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
        spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

        # Combine features
        features = np.hstack([mfccs, rmse, spectral_flux, zcr])
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None


def test_new_audio(file_path):
    # Load the saved model and scaler
    model = joblib.load('best_svm_model.joblib')
    scaler = joblib.load('scaler.joblib')

    # Extract features from the new audio file
    features = extract_features(file_path)

    if features is not None:
        # Scale the features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        classes = {0:"Low Fluency", 1:"Intermediate Fluency", 2:"High Fluency"}

        print(f"Predicted class for {file_path}: {classes.get(prediction[0])}")
    else:
        print("Failed to extract features from the audio file.")


# Main execution
if __name__ == "__main__":
    # Test a new audio file
    new_audio_file = "pitch-audio.wav"
    test_new_audio(new_audio_file)
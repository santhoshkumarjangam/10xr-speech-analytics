import glob
import os
import librosa
import numpy as np
import soundfile as sf


def feature_extraction(file_name):
    try:
        # Use soundfile to read the audio file
        X, sample_rate = sf.read(file_name)

        if X.ndim > 1:
            X = X[:, 0]  # If stereo, take only the first channel

        # Proceed with feature extraction
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
        spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

        return mfccs, rmse, spectral_flux, zcr
    except Exception as e:
        print(f"Error extracting features from {file_name}: {str(e)}")
        return None


def parse_audio_files(parent_dir, sub_dirs, file_ext='*'):  # Changed to '*' to match all files
    n_mfccs = 20
    number_of_features = 3 + n_mfccs
    features, labels = [], []

    for label, sub_dir in enumerate(sub_dirs):
        sub_dir_path = os.path.join(parent_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            print(f"Warning: {sub_dir_path} is not a directory. Skipping.")
            continue

        for file_name in glob.glob(os.path.join(sub_dir_path, file_ext)):
            result = feature_extraction(file_name)
            if result is not None:
                mfccs, rmse, spectral_flux, zcr = result
                extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
                features.append(extracted_features)
                labels.append(label)

        print(f"Extracted features from {sub_dir}, processed {len(features) - sum(labels[:-1]) if labels else 0} files")

    return np.array(features), np.array(labels, dtype=np.int64)


# Main execution
parent_dir = "audio files"
if not os.path.isdir(parent_dir):
    print(f"Error: {parent_dir} is not a directory or doesn't exist.")
    exit(1)

audio_subdirectories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
audio_subdirectories.sort()
print('Audio Subdirs: ', audio_subdirectories)

if not audio_subdirectories:
    print("Error: No subdirectories found in 'audio files' directory.")
    exit(1)

features, labels = parse_audio_files(parent_dir, audio_subdirectories)
print(f"Total extracted features: {len(features)}")
print(f"Total labels: {len(labels)}")

if len(features) > 0 and len(labels) > 0:
    np.save('feat.npy', features)
    np.save('label.npy', labels)
    print("Features and labels saved successfully.")
else:
    print("No features or labels extracted. Check your audio files and directory structure.")
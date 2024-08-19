import librosa
import numpy as np


def analyze_audio_energy(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract features
    energy = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Calculate overall energy metrics
    avg_energy = np.mean(energy)
    avg_zcr = np.mean(zcr)
    avg_spectral_centroid = np.mean(spectral_centroid)

    # Simple energy score calculation
    energy_score = (avg_energy + avg_zcr + avg_spectral_centroid) / 3

    return energy_score, avg_energy, avg_zcr, avg_spectral_centroid



file_path = "pitch-audio.wav"
energy_score, avg_energy, avg_zcr, avg_spectral_centroid = analyze_audio_energy(file_path)

print(f"Overall Energy Score: {energy_score}")
print(f"Average Energy: {avg_energy}")
print(f"Average Zero Crossing Rate: {avg_zcr}")
print(f"Average Spectral Centroid: {avg_spectral_centroid}")
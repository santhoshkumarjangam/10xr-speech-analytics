import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from scipy.stats import skew, kurtosis


class SpeakerEnthusiasmEvaluator:
    def __init__(self):
        # Load pre-trained model for speech emotion recognition
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.model = AutoModelForAudioClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

    def extract_prosodic_features(self, y, sr):
        # Extract pitch (F0) and pitch statistics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        pitch_stats = {
            'mean': np.mean(pitches),
            'std': np.std(pitches),
            'skew': skew(pitches),
            'kurtosis': kurtosis(pitches)
        }

        # Extract speech rate (using zero-crossing rate as a proxy)
        zcr = librosa.feature.zero_crossing_rate(y)
        speech_rate = np.mean(zcr)

        # Extract rhythm (using onset strength)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        rhythm = np.mean(onset_env)

        return np.array([
            pitch_stats['mean'], pitch_stats['std'], pitch_stats['skew'],
            pitch_stats['kurtosis'], speech_rate, rhythm
        ])

    def extract_spectral_features(self, y, sr):
        # Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_stats = np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_stats = np.hstack([np.mean(contrast, axis=1), np.std(contrast, axis=1)])

        return np.hstack([mfcc_stats, contrast_stats])

    def predict_emotion(self, y, sr):
        # Resample to 16kHz for the pre-trained model
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Extract features
        inputs = self.feature_extractor(y_resampled, sampling_rate=16000, return_tensors="pt", padding=True)

        # Get model prediction
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class_ids = torch.argmax(logits, dim=-1).item()

        # Map class ID to emotion label
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

        # Check if the predicted_class_ids are within the expected range
        if predicted_class_ids < len(emotions):
            return emotions[predicted_class_ids]
        else:
            return "unknown"  # Handle unexpected class IDs

    def evaluate_enthusiasm(self, audio_file):
        # Load the audio file
        y, sr = librosa.load(audio_file)

        # Extract features
        prosodic_features = self.extract_prosodic_features(y, sr)
        spectral_features = self.extract_spectral_features(y, sr)
        emotion = self.predict_emotion(y, sr)

        # Combine features
        features = np.hstack([prosodic_features, spectral_features])

        # Calculate enthusiasm score
        # This is a simplified heuristic and should be replaced with a trained model
        pitch_variation = features[1]  # Pitch standard deviation
        speech_rate = features[4]
        rhythm = features[5]

        base_score = (pitch_variation + speech_rate + rhythm) / 3

        # Adjust score based on emotion
        emotion_multiplier = {
            'happy': 1.2,
            'angry': 1.1,
            'fear': 0.9,
            'neutral': 1.0,
            'sad': 0.8,
            'disgust': 0.9,
            'ps': 1.0  # 'ps' might stand for 'pleasant surprise'
        }

        enthusiasm_score = base_score * emotion_multiplier[emotion]

        # Normalize to 0-100 scale
        normalized_score = int(min(100, max(0, enthusiasm_score * 50)))

        return normalized_score, emotion


# Usage
evaluator = SpeakerEnthusiasmEvaluator()
score, emotion = evaluator.evaluate_enthusiasm("pitch-audio.wav")
print(f"Speaker Enthusiasm Score: {score}/100")
print(f"Detected Emotion: {emotion}")
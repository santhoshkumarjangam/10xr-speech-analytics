import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from typing import Tuple, List


class SpeechClarityEvaluator:
    def __init__(self):
        # Load pre-trained wav2vec model for speech recognition
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def preprocess_audio(self, audio_file: str) -> np.ndarray:
        """Preprocess audio file to match model requirements."""
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio_array = np.array(audio.get_array_of_samples())
        return audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    def extract_acoustic_features(self, y: np.ndarray, sr: int) -> dict:
        """Extract acoustic features related to speech clarity."""
        # Spectral centroid (related to speech "brightness")
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # Spectral flatness (measure of noisiness)
        flat = librosa.feature.spectral_flatness(y=y)

        # Spectral bandwidth (related to speech "spread")
        band = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # Spectral contrast (related to voice/unvoiced separation)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return {
            "centroid_mean": np.mean(cent),
            "centroid_std": np.std(cent),
            "flatness_mean": np.mean(flat),
            "flatness_std": np.std(flat),
            "bandwidth_mean": np.mean(band),
            "bandwidth_std": np.std(band),
            "contrast_mean": np.mean(contrast),
            "contrast_std": np.std(contrast),
            "mfcc_mean": np.mean(mfcc, axis=1),
            "mfcc_std": np.std(mfcc, axis=1)
        }

    def compute_speech_rate(self, audio_array: np.ndarray) -> float:
        """Compute speech rate using wav2vec model."""
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits  # Removed attention_mask here
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        word_count = len(transcription.split())
        duration = len(audio_array) / 16000  # in seconds
        return word_count / duration

    def evaluate_clarity(self, audio_file: str) -> Tuple[float, dict]:
        """Evaluate speech clarity from audio file."""
        # Preprocess audio
        audio_array = self.preprocess_audio(audio_file)

        # Extract acoustic features
        features = self.extract_acoustic_features(audio_array, sr=16000)

        # Compute speech rate
        speech_rate = self.compute_speech_rate(audio_array)
        features['speech_rate'] = speech_rate

        # Compute clarity score (this is a simplified heuristic and should be replaced with a trained model)
        clarity_score = (
                                (1 - features['flatness_mean']) * 0.3 +  # Less flatness is clearer
                                features['contrast_mean'] * 0.3 +  # More contrast is clearer
                                (1 - abs(features['speech_rate'] - 3)) * 0.4
                        # Optimal speech rate around 3 words/second
                        ) * 100  # Scale to 0-100

        return clarity_score, features

    def evaluate_multiple_speakers(self, audio_files: List[str]) -> List[Tuple[str, float, dict]]:
        """Evaluate speech clarity for multiple audio files."""
        results = []
        for audio_file in audio_files:
            clarity_score, features = self.evaluate_clarity(audio_file)
            results.append((audio_file, clarity_score, features))
        return results


# Usage example
evaluator = SpeechClarityEvaluator()

# Single speaker evaluation
audio_file = "pitch-audio.wav"
clarity_score, features = evaluator.evaluate_clarity(audio_file)
print(f"Speech Clarity Score: {clarity_score:.2f}")
print("Acoustic Features:")
for feature, value in features.items():
    print(f"   {feature}: {value}")

# Multiple speaker evaluation
"""
audio_files = ["speaker1.wav", "speaker2.wav", "speaker3.wav"]
results = evaluator.evaluate_multiple_speakers(audio_files)
for audio_file, clarity_score, features in results:
    print(f"\nAudio File: {audio_file}")
    print(f"Speech Clarity Score: {clarity_score:.2f}")
    print("Key Acoustic Features:")
    print(f"  Spectral Flatness: {features['flatness_mean']:.4f}")
    print(f"  Spectral Contrast: {features['contrast_mean']:.4f}")
    print(f"  Speech Rate: {features['speech_rate']:.2f} words/second")
"""
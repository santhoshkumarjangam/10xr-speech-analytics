from transformers import pipeline
import librosa

import os
import sys
torch_path = os.path.dirname(sys.executable)
os.environ['PATH'] = f"{torch_path};{os.environ['PATH']}"

def analyze_emotion(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Initialize emotion recognition pipeline
    classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

    # Perform emotion recognition
    result = classifier(file_path)

    return result


# Example usage
file_path = "pitch-audio.wav"
emotion_result = analyze_emotion(file_path)
print(emotion_result)
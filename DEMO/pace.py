import matplotlib.pyplot as plt
import wave
from transcription import transcribe_audio


def get_audio_duration(audio_file):
    # For .wav files, use the wave module
    with wave.open(audio_file, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration


def calculate_wpm(transcription, duration_seconds):
    word_count = len(transcription.split())
    duration_minutes = duration_seconds / 60.0
    wpm = word_count / duration_minutes
    return wpm, word_count

def normalize_wpm(wpm, min_wpm=80, max_wpm=200):
    """
    Normalize the WPM to a 0-10 scale.
    """
    if wpm < min_wpm:
        return 0
    elif wpm > max_wpm:
        return 10
    else:
        return (wpm - min_wpm) / (max_wpm - min_wpm) * 10

def plot_pace_graph(wpm, pace_score):
    """
    Plot the pace score on a graph.
    """
    plt.figure(figsize=(8, 4))
    plt.bar(["Pace"], [pace_score], color='blue')
    plt.ylim(0, 10)
    plt.ylabel('Pace (0-10 Scale)')
    plt.title(f"Pace Score: {pace_score:.2f} (WPM: {wpm:.2f})")
    plt.show()

if __name__ == "__main__":
    audio_file = "pitch-audio.wav"

    # Get audio duration
    duration = get_audio_duration(audio_file)
    #duration = 18

    # Transcribe audio
    transcript = transcribe_audio(audio_file)
    #transcript = ("hey there, um i am santhosh, i am really excited to work on this project i mean its pretty much enjoyable and i am learning a lot. um so i guess that is it this is the end of the you know the transcript")
    if transcript.startswith("Could not"):
        print(transcript)
    else:
        # Calculate WPM
        wpm, word_count = calculate_wpm(transcript, duration)
        print(f"Words Per Minute (WPM): {wpm:.2f}")
        print(f"Total Words: {word_count}")
        print(f"Audio Duration (Seconds): {duration:.2f}")
        pace_score = normalize_wpm(wpm)
        print("pace score :",pace_score)
        plot_pace_graph(wpm, pace_score)

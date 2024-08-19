from transcription import transcribe_audio

def count_filler_words(transcript, filler_words):
    word_list = transcript.lower().split()
    fillerCount = {}
    for word in filler_words:
        fillerCount[word] = word_list.count(word)
    return fillerCount

if __name__ == "__main__":
    audioFile = "pitch-audio.wav"
    filler_words = ["uh", "um", "ah", "you-know", "so", "actually", "basically", "well","Hmm","er"]
    transcript = transcribe_audio(audioFile)
    print(transcript)
    #transcript = ("hey there, um i am santhosh, i am really excited to work on this project well its pretty much enjoyable and i am learning a lot. um so i guess that is it this is the end of the you-know the transcript")

    if transcript.startswith("Could not"):
        print(transcript)
    else:
        filler_count = count_filler_words(transcript, filler_words)
        print("Filler Word Counts:", filler_count)
        print("Filler Words Used:", [word for word, count in filler_count.items() if count > 0])

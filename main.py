import whisper

model = whisper.load_model("base")
result = model.transcribe("pitch-audio.wav")
print(result["text"])

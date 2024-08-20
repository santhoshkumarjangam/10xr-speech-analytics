import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import language_tool_python
import numpy as np
from scipy.stats import percentileofscore


class LanguageProficiencyEvaluator:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.fluency_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.coherence_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.coherence_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    def transcribe_audio(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""

    def assess_fluency(self, text):
        result = self.fluency_model(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']

    def assess_grammar(self, text):
        errors = self.grammar_tool.check(text)
        return 1 - (len(errors) / len(text.split()))  # Simplified metric

    def assess_coherence(self, text):
        inputs = self.coherence_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.coherence_model(**inputs)
        return outputs.logits.softmax(dim=1)[0][1].item()  # Assuming binary classification

    def assess_vocabulary(self, text):
        # Simplified metric: unique words / total words
        words = text.lower().split()
        return len(set(words)) / len(words)

    def evaluate_proficiency(self, audio_file):
        text = self.transcribe_audio(audio_file)
        if not text:
            return 0  # Unable to transcribe

        fluency_score = self.assess_fluency(text)
        grammar_score = self.assess_grammar(text)
        coherence_score = self.assess_coherence(text)
        vocabulary_score = self.assess_vocabulary(text)

        # Combine scores (you may want to adjust weights based on importance)
        combined_score = np.mean([fluency_score, grammar_score, coherence_score, vocabulary_score])

        # Convert to percentile (assuming a normal distribution of scores)
        percentile = percentileofscore(np.random.normal(0.5, 0.15, 1000), combined_score)

        # Map percentile to 1-100 scale
        return max(1, min(100, round(percentile)))


# Usage
evaluator = LanguageProficiencyEvaluator()
score = evaluator.evaluate_proficiency("pitch-audio.wav")
print(f"Language Proficiency Score: {score}/100")
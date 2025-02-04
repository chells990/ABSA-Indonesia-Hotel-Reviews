import re
import joblib


class TextPreprocessor:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.punctuation = re.compile(r'[^\w\s]')

    def clean_text(self, text, for_bert=False):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\d+', '', text)      # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip()

        if not for_bert:
            text = self.punctuation.sub('', text)
            text = ' '.join([w for w in text.split() if w not in self.stopwords])
        return text

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
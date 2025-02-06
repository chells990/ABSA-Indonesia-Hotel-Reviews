from pathlib import Path
from transformers import AutoTokenizer
from bert_architecture import ABSABERT
from svm_architecture import ABSASVM
import torch


class ABSAInferencer:
    def __init__(self, model_dir, aspects, preprocessor, model_type='bert'):
        self.aspects = aspects
        self.preprocessor = preprocessor
        self.model_type = model_type
        self.label_map = {0: 'neg', 1: 'neut', 2: 'pos'}

        if model_type == 'bert':
            self.model = ABSABERT.load(Path(model_dir), aspects)
            self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir))
        else:
            self.model = ABSASVM.load(Path(model_dir), aspects)

    def predict(self, text):
        if self.model_type == 'bert':
            return self._predict_bert(text)
        return self._predict_svm(text)

    def _predict_bert(self, text):
        clean_text = self.preprocessor.clean_text(text, for_bert=True)
        encoding = self.tokenizer.encode_plus(
            clean_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Move input tensors to the model's device
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

        preds = {}
        for aspect in self.aspects:
            with torch.no_grad():
                outputs = self.model.models[aspect](**encoding)
            preds[aspect] = self.label_map[torch.argmax(outputs.logits).item()]
        return preds

    def _predict_svm(self, text):
        clean_text = self.preprocessor.clean_text(text, for_bert=False)
        X = self.model.vectorizer.transform([clean_text])
        return {
            aspect: self.label_map[self.model.models[aspect].predict(X)[0]]
            for aspect in self.aspects
        }
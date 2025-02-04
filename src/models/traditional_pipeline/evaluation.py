from pathlib import Path
from src.models.traditional_pipeline.training import load_and_convert_data
from src.models.traditional_pipeline.text_preprocessor import TextPreprocessor
from src.models.traditional_pipeline.inference import ABSAInferencer




if __name__ == "__main__":
    aspects = ['ac', 'air_panas', 'bau', 'general', 'kebersihan',
        'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    df_test = load_and_convert_data('test_preprocess.csv', aspects)

    # Load inferencers
    model_dir = Path("saved_models")
    preprocessor = TextPreprocessor.load(model_dir/"preprocessor.joblib")
    svm_infer = ABSAInferencer(model_dir/"svm", aspects, preprocessor, 'svm')
    bert_infer = ABSAInferencer(model_dir/"bert", aspects, preprocessor, 'bert')
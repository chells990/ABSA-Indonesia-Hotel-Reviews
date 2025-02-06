from pathlib import Path
from training import load_and_convert_data
from preprocess import TextPreprocessor
from inferencer import ABSAInferencer




if __name__ == "__main__":
    aspects = ['ac', 'air_panas', 'bau', 'general', 'kebersihan',
        'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    df_test = load_and_convert_data('source_data/test_preprocess.csv', aspects)
    text = '"lumayan nyaman,tp kebersihan kmr mandi perlu ditingkatkan lg biar gk ada kuning2 di sudutnya lbh bgs"'
    # Load inferencers
    model_dir = Path("saved_models")
    preprocessor = TextPreprocessor.load(model_dir/"preprocessor.joblib")
    svm_infer = ABSAInferencer(model_dir/"svm", aspects, preprocessor, 'svm')
    bert_infer = ABSAInferencer(model_dir/"bert", aspects, preprocessor, 'bert')
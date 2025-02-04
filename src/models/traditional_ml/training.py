import pandas as pd
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.traditional_ml.absa_svm import ABSASVM
from src.models.deep_learning.absa_bert import ABSABERT
from src.models.traditional_ml.text_preprocessor import TextPreprocessor
from src.models.deep_learning.absa_dataset import ABSADataset
import warnings

nltk.download('stopwords')
warnings.filterwarnings("ignore")




def load_and_convert_data(file_path, aspect_cols):
    """Load CSV and convert sentiment labels to integers"""
    df = pd.read_csv(file_path)
    label_map = {'neg': 0, 'neut': 1, 'pos': 2}

    for col in aspect_cols:
        df[col] = df[col].map(label_map).fillna(1).astype('int8')  # neut as default

    return df


# Training
if __name__ == "__main__":

    aspects = ['ac', 'air_panas', 'bau', 'general', 'kebersihan',
             'linen', 'service', 'sunrise_meal', 'tv', 'wifi']

    df_train = load_and_convert_data('train_preprocess.csv', aspects)
    df_val = load_and_convert_data('valid_preprocess.csv', aspects)

    # Initialize components
    stopwords_id = stopwords.words('indonesian')
    preprocessor = TextPreprocessor(stopwords_id)

    # Preprocess text
    for df in [df_train, df_val]:
        df['clean_svm'] = df['review'].apply(preprocessor.clean_text, for_bert=False)
        df['clean_bert'] = df['review'].apply(preprocessor.clean_text, for_bert=True)

    # Train SVM
    svm_model = ABSASVM(aspects)
    svm_model.train(
        df_train['clean_svm'],
        {a: df_train[a] for a in aspects},
        df_val['clean_svm'],
        {a: df_val[a] for a in aspects}
    )

    # Train BERT
    bert_model = ABSABERT(aspects)
    train_loaders = {
        a: DataLoader(
            ABSADataset(df_train['clean_bert'], df_train[a], bert_model.tokenizer),
            batch_size=16,
            shuffle=True
        ) for a in aspects
    }
    val_loaders = {
        a: DataLoader(
            ABSADataset(df_val['clean_bert'], df_val[a], bert_model.tokenizer),
            batch_size=16
        ) for a in aspects
    }
    bert_model.train(train_loaders, val_loaders)

    # Save models
    model_dir = Path("saved_models")
    svm_model.save(model_dir/"svm")
    bert_model.save(model_dir/"bert")
    preprocessor.save(model_dir/"preprocessor.joblib")
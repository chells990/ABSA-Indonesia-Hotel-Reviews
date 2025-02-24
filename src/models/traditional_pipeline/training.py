import pandas as pd
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from torch.utils.data import DataLoader
import time
import warnings
from svm_architecture import ABSASVM
from bert_architecture import ABSABERT
from preprocess import TextPreprocessor
from bert_dataset import ABSADataset

nltk.download('stopwords')
warnings.filterwarnings("ignore")


def load_and_convert_data(file_path, aspect_cols):
    """Load CSV and convert sentiment labels to integers"""
    df = pd.read_csv(file_path)
    label_map = {'neg': 0, 'pos': 2}

    for col in aspect_cols:
        df[col] = df[col].map(label_map).fillna(1) # 1 for neutral
        df[col] = df[col].astype('int8')
    return df



if __name__ == "__main__":

    aspects = ['ac', 'air_panas', 'bau', 'general', 'kebersihan',
               'linen', 'service', 'sunrise_meal', 'tv', 'wifi']

    # Load data
    df_train = load_and_convert_data('source_data/train_preprocess.csv', aspects)
    df_val = load_and_convert_data('source_data/valid_preprocess.csv', aspects)

    # Initialize preprocessor using Indonesian stopwords
    stopwords_id = stopwords.words('indonesian')
    preprocessor = TextPreprocessor(stopwords_id)

    # Preprocess text
    for df in [df_train, df_val]:
        df['clean_svm'] = df['review'].apply(lambda x: preprocessor.clean_text(x, for_bert=False))
        df['clean_bert'] = df['review'].apply(lambda x: preprocessor.clean_text(x, for_bert=True))

    print("Training SVM model...")
    start_time = time.time()
    svm_model = ABSASVM(aspects)
    svm_model.train(
        df_train['clean_svm'],
        {a: df_train[a] for a in aspects},
        df_val['clean_svm'],
        {a: df_val[a] for a in aspects}
    )
    end_time = time.time()
    svm_training_time = end_time - start_time
    

    print("Training BERT model...")
    start_time = time.time()
    bert_model = ABSABERT(aspects)
    batch_size = 32
    train_loaders = {
        a: DataLoader(
            ABSADataset(df_train['clean_bert'], df_train[a], bert_model.tokenizer),
            batch_size=batch_size,
            shuffle=True
        ) for a in aspects
    }
    val_loaders = {
        a: DataLoader(
            ABSADataset(df_val['clean_bert'], df_val[a], bert_model.tokenizer),
            batch_size=batch_size
        ) for a in aspects
    }

    bert_model.train(train_loaders, val_loaders)
    end_time = time.time()
    bert_training_time = end_time - start_time

    print(f"SVM training completed in {svm_training_time:.3f} seconds.")
    print(f"BERT training completed in {bert_training_time:.3f} seconds.")

    model_dir = Path("saved_models")
    svm_model.save(model_dir / "svm")
    bert_model.save(model_dir / "bert")
    preprocessor.save(model_dir / "preprocessor.joblib")
    
    print("\nModels and preprocessor saved successfully.")
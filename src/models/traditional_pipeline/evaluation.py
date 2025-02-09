import time
import pandas as pd
from pathlib import Path
from training import load_and_convert_data
from preprocess import TextPreprocessor
from inferencer import ABSAInferencer


def create_pred_df(inferencer, reviews):
    predictions = []
    total_time = 0
    for review in reviews:
        start_time = time.time()  
        pred = inferencer.predict(review)
        end_time = time.time() 
        inference_time = end_time - start_time 
        total_time += inference_time 
        pred['review'] = review
        predictions.append(pred)
    
    df = pd.DataFrame(predictions)
    avg_time_per_review = total_time / len(reviews)

    print(f"Total inference time for {inferencer.model_type}: {total_time:.4f} seconds")
    print(f"Average inference time per review for {inferencer.model_type}: {avg_time_per_review:.4f} seconds")
    return df[['review'] + aspects]



if __name__ == "__main__":
    aspects = ['ac', 'air_panas', 'bau', 'general', 'kebersihan',
               'linen', 'service', 'sunrise_meal', 'tv', 'wifi']
    
    # Load test data
    df_test = load_and_convert_data('source_data/test_preprocess.csv', aspects)
    
    # Load inferencers
    model_dir = Path("saved_models")
    preprocessor = TextPreprocessor.load(model_dir/"preprocessor.joblib")
    svm_infer = ABSAInferencer(model_dir/"svm", aspects, preprocessor, 'svm')
    bert_infer = ABSAInferencer(model_dir/"bert", aspects, preprocessor, 'bert')

    # Inference
    print("Running SVM inference...")
    svm_df = create_pred_df(svm_infer, df_test['review'])
    print("\nRunning BERT inference...")
    bert_df = create_pred_df(bert_infer, df_test['review'])

    # Save predictions to CSV
    eval_dir = Path("evaluation")
    svm_df.to_csv(eval_dir/"svm_predictions.csv", index=False)
    bert_df.to_csv(eval_dir/"bert_predictions.csv", index=False)
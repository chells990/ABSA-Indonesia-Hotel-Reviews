import pandas as pd
import json
from constant import ID_HOTEL_REVIEW_ABSA_PROMPT
import os
from dotenv import load_dotenv
load_dotenv()

# API keys (if needed)
gpt_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_API_KEY')
claude_key = os.getenv('ANTHROPONIC_API_KEY')

def batching_review(df, n):
    """
    Load CSV and group the 'review' column into batches of 10.
    
    Returns:
        dict: A dictionary where each key is a batch ID (e.g., "batch_1") and 
              each value is a list of up to 10 reviews.
    """
    reviews = df.tolist()
    
    batches = {}
    for i in range(0, len(reviews), n):
        batch_id = f"batch_{(i // n) + 1}"
        batches[batch_id] = reviews[i:i+n]
    
    return batches

if __name__ == "__main__":
    # Load and batch the reviews
    test_df = pd.read_csv('source_data/test_preprocess.csv')
    df_review = test_df['review']
    batched_reviews = batching_review(df_review, n = 10)
    
    for batch_id, reviews in batched_reviews.items():
        batch_texts = { f"text_{i}": review for i, review in enumerate(reviews, start=1) }
        input_dict = { batch_id: batch_texts }
        formatted_prompt = ID_HOTEL_REVIEW_ABSA_PROMPT.format(
            input_variable=json.dumps(input_dict, ensure_ascii=False, indent=2)
        )
        
        print("Prompt for", batch_id)
        print(formatted_prompt)
        print("-" * 80)

import pandas as pd
import json
import time  # For timing
from pathlib import Path
from openai import OpenAI
from constant import FEW_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT, ZERO_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def batching_review(reviews, n):
    batches = {}
    for i in range(0, len(reviews), n):
        batch_id = f"batch_{(i // n) + 1}"
        batches[batch_id] = reviews[i:i+n]
    return batches

def process_batch_response(batch_id, batch_response, original_batch, aspect_keys=None):
    if aspect_keys is None:
        aspect_keys = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service", "sunrise_meal", "tv", "wifi"]

    rows = []
    for review in original_batch:
        aspects = batch_response.get(batch_id, {}).get(review, {})
        row = {"batch_id": batch_id, "review": review}
        for key in aspect_keys:
            row[key] = aspects.get(key, 99)
        rows.append(row)
    return rows

def call_openai_api(model, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in aspect-based sentiment analysis for hotel reviews written in Indonesian. Please respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0
        )
        response_text = response.choices[0].message.content.strip()
        #print(f"Raw response from {model}: '{response_text}'")
        print(f"'{model}' -> Converting the batch response to JSON format...")
        return json.loads(response_text)
    except Exception as e:
        print(f"Error during API call or JSON parsing for {model}: {e}")
        return {}


if __name__ == "__main__":
    # Set prompt_mode to "zero" or "few"
    prompt_mode = "few"
    if prompt_mode == "zero":
        prompt_template = ZERO_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT
        print("Using ZERO-SHOT prompt")
    else:
        prompt_template = FEW_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT
        print("Using FEW-SHOT prompt")

    aspect_keys = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service", "sunrise_meal", "tv", "wifi"]
    test_df = pd.read_csv('source_data/test_preprocess.csv')
    test_df["review_trimmed"] = test_df["review"].str.strip()
    test_df = test_df.drop(columns=aspect_keys, errors="ignore")
    reviews = test_df["review_trimmed"].tolist()
    batched_reviews = batching_review(reviews, n=10)

    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    label_map = {0: 'neg', 1: 'pos', 99: 'neut'}
    models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"] # Add more models here

    for model in models:
        print(f"\nProcessing model: {model}")
        start_model = time.time()  # Start timer for the entire model processing
        all_rows = []
        
        for batch_id, original_batch in batched_reviews.items():
            formatted_prompt = prompt_template.format(
                input_variable=json.dumps({batch_id: original_batch}, ensure_ascii=False, indent=2)
            )
            batch_response = call_openai_api(model, formatted_prompt)
            batch_rows = process_batch_response(batch_id, batch_response, original_batch, aspect_keys)
            all_rows.extend(batch_rows)
        
        end_model = time.time()  # End timer after processing all batches
        total_model_time = end_model - start_model
        total_reviews = len(reviews)
        avg_review_time = total_model_time / total_reviews if total_reviews > 0 else 0
        
        df_final = pd.DataFrame(all_rows)
        df_final.drop(columns=["batch_id"], inplace=True)
        for col in aspect_keys:
            df_final[col] = df_final[col].map(label_map)
        
        result_df = test_df.merge(df_final, left_on="review_trimmed", right_on="review", how="left").drop(columns=["review_trimmed", "review_y"])
        result_df = result_df.rename(columns={"review_x": "review"})
        result_df.to_csv(eval_dir / f"{prompt_mode}_shot_{model}_predictions.csv", index=False)

        print(f"Predictions saved for {model} at {eval_dir}")
        print(f"Total inference time for {model}: {total_model_time:.4f} seconds")
        print(f"Average inference time per review for {model}: {avg_review_time:.4f} seconds")
        print(f"Completed processing for {model}")
import pandas as pd
import json
# import openai
from constant import ID_HOTEL_REVIEW_ABSA_PROMPT, batch_response_test
import os
from dotenv import load_dotenv

load_dotenv()

#openai.api_key = os.getenv('OPENAI_API_KEY')

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
            # Directly get the aspect value (no nested "label" key)
            row[key] = aspects.get(key, 99)  # Use 99 as the default value
        rows.append(row)
    return rows

# def call_openai_api(prompt):
#     try:
#         response = openai.Completion.create(
#             model="text-davinci-003",
#             prompt=prompt,
#             max_tokens=1500,
#             temperature=0
#         )
#         response_text = response.choices[0].text.strip()
#         return json.loads(response_text)
#     except Exception as e:
#         print(f"Error during API call or JSON parsing: {e}")
#         return {}


if __name__ == "__main__":
    aspect_keys = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service", "sunrise_meal", "tv", "wifi"]
    test_df = pd.read_csv('source_data/test_preprocess.csv')
    df_review = test_df['review'].head(20)
    batched_reviews = batching_review(df_review.tolist(), n=10)

    all_rows = []
    for batch_id, original_batch in batched_reviews.items():
        formatted_prompt = ID_HOTEL_REVIEW_ABSA_PROMPT.format(
            input_variable=json.dumps({batch_id: original_batch}, ensure_ascii=False, indent=2)
        )
        # print(formatted_prompt)
        # print("-" * 100)

        # batch_response = call_openai_api(formatted_prompt)
        # before go with openai, let's use the batch_response_test
        batch_response = batch_response_test.strip()
        batch_response = json.loads(batch_response)

        batch_rows = process_batch_response(batch_id, batch_response, original_batch, aspect_keys)
        all_rows.extend(batch_rows)  

    df_final = pd.DataFrame(all_rows)
    columns_order = ["batch_id", "review"] + aspect_keys
    df_final = df_final[columns_order]
    df_final.drop(columns=["batch_id"], inplace=True)
    label_map = {0: 'neg', 1 : 'neut', 99 : 'neut'}

    for col in aspect_keys:
        df_final[col] = df_final[col].map(label_map)

    print(df_final)
    df_final.to_csv('output_test.csv', index=False)

import pandas as pd
import os
from dotenv import load_dotenv 
load_dotenv() 


# api key
gpt_key = os.getenv('OPENAI_API_KEY')
gemini_key = os.getenv('GOOGLE_API_KEY')
claude_key = os.getenv('ANTHROPONIC_API_KEY')

# prompt
ID_HOTEL_REVIEW_ABSA_PROMPT = """
You are an expert in aspect based sentiment analysis for hotel reviews written in Indonesian. Your role is to analyze review texts and extract clear positive or negative sentiments for specific aspects. If an aspect is ambiguous or not clearly expressed, do not output it.

Aspects Consideration:
Evaluate the following aspects:
- Air Conditioner (AC): Comments on AC performance or issues.
- Air Panas: Remarks regarding water temperature.
- Bau: Observations on unpleasant odors.
- General: Overall impressions of the review.
- Kebersihan: Feedback on the hotel's cleanliness.
- Linen: Opinions on bedding quality.
- Service: Evaluations of the service provided.
- Sunrise Meal: Comments on the breakfast experience.
- Television (TV): Remarks on TV quality or functionality.
- WIFI: Statements regarding internet connectivity.
The shorthand keys for these aspects are: ['ac', 'air_panas', 'bau', 'general', 'kebersihan', 'linen', 'service', 'sunrise_meal', 'tv', 'wifi'].

[INPUT FORMAT]
The input will be a dictionary with the following format:
{{ "batch_id": {{"text_1": \"review_1\", "text_2": \"review_2\", ..., "text_10": \"review_10\" }} }}

[OUTPUT FORMAT]
The output dict should have the following format:
 {{"batch_id": {{ \"review_1\": {{ "key_sentiment_aspect_1": 0 or 1, "key_sentiment_aspect_2": 0 or 1, ..., "key_sentiment_aspect_n": 0 or 1 }}, \"review_2\": {{ ... }}, ..., \"review_10\": {{ ... }} }} }}

[NOTES]
- Expect inconsistencies, typos, and variations in the data. Your model should be capable of handling these and determining the most likely values.
- For each aspect, assign a sentiment: positive is represented by 1 and negative by 0.
- Output a JSON object for each review with keys and corresponding sentiment.
- Do not output keys for aspects with ambiguous sentiments or when the sentiment is not clearly positive or negative.
- Sometimes the review will be written in mixed languagues. You should be able to handle that as well and map it to the possible values in the aspects. You should approach it by translating the original text to indonesian first, then do the labelling.

[EXAMPLES]
input: "batch_id": {{ {{ "text_1": \"AC nya sangat bagus, tapi WIFI sering mati.\", "text_2": \"Pelayanannya memuaskan dan sarapan sangat lezat, meski linen kurang bagus.\" }} }}
output: "batch_id": {{ {{ \"AC nya sangat bagus, tapi WIFI sering mati.\": {{ "ac": 1, "wifi": 0 }}, \"Pelayanannya memuaskan dan sarapan sangat lezat, meski linen kurang bagus.\": {{ "service": 1, "sunrise_meal": 1, "linen": 0 }} }} }}

[ABSA TASK]
INPUT: {input_variable}
OUTPUT:
"""
import pandas as pd
import os
from dotenv import load_dotenv 
load_dotenv() 


# api key
gpt_key = os.getenv('OPENAI_API_KEY')

# prompt
FEW_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT = """
Your role is to analyze review texts and extract clear positive or negative sentiments for specific aspects. If an aspect is ambiguous or not clearly expressed, do not output it.

Aspects Consideration:
Evaluate the following aspects:
- Air Conditioner (AC): Comments on AC performance or issues.
- Air Panas: Remarks regarding water temperature.
- Bau: Observations on unpleasant odors.
- General: Overall impressions of the review.
- Kebersihan: Feedback on the hotel's cleanliness.
- Linen: Opinions on bedding quality also include complimentary item like free snack and soap.
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
- Output a valid JSON object for each review with keys and corresponding sentiment without any markdown formatting or code fences.
- Do not output keys for aspects with ambiguous sentiments or when the sentiment is not clearly positive or negative.
- Sometimes the review will be written in mixed languagues. You should be able to handle that as well and map it to the possible values in the aspects. You should approach it by translating the original text to indonesian first, then do the labelling.

[EXAMPLES]
input: "batch_id": {{ {{ "text_1": \"AC nya sangat bagus, tapi WIFI sering mati.\", "text_2": \"Pelayanannya memuaskan dan sarapan sangat lezat, meski linen kurang bagus.\", "text_3": \""secara umum bagus, cuma sprei, sarung bantal dan handuk tidak layak...maaf.."\", "text_4": \"Kamar bersih dan nyaman tetapi amenties dari airy tidak disediakan. Begitu juga dengan compliment free snack dari airy\" }} }}
output: "batch_id": {{ {{ \"AC nya sangat bagus, tapi WIFI sering mati.\": {{ "ac": 1, "wifi": 0 }}, \"Pelayanannya memuaskan dan sarapan sangat lezat, meski linen kurang bagus.\": {{ "service": 1, "sunrise_meal": 1, "linen": 0 }}, \"kamar mandi nya sedikit bau, selebihnya memuaskan.\": {{ "bau": 0, "general": 1}}, \"Petugasnya ramah banget., tapi kelengkapan kurang., tidak ada air panas., makanan kompliment juga tidak ada..\": {{ "air_panas": 0, "linen": 0, "service": 1 }} }} }}

[ABSA TASK]
INPUT: {input_variable}
OUTPUT:
"""

ZERO_SHOT_ID_HOTEL_REVIEW_ABSA_PROMPT = """
Your role is to analyze review texts and extract clear positive or negative sentiments for specific aspects. If an aspect is ambiguous or not clearly expressed, do not output it.

Aspects Consideration:
Evaluate the following aspects:
- Air Conditioner (AC): Comments on AC performance or issues.
- Air Panas: Remarks regarding water temperature.
- Bau: Observations on unpleasant odors.
- General: Overall impressions of the review.
- Kebersihan: Feedback on the hotel's cleanliness.
- Linen: Opinions on bedding quality also include complimentary item like free snack and soap.
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
- Output a valid JSON object for each review with keys and corresponding sentiment without any markdown formatting or code fences.
- Do not output keys for aspects with ambiguous sentiments or when the sentiment is not clearly positive or negative.
- Sometimes the review will be written in mixed languagues. You should be able to handle that as well and map it to the possible values in the aspects. You should approach it by translating the original text to indonesian first, then do the labelling.

[ABSA TASK]
INPUT: {input_variable}
OUTPUT:
"""

# test purposes
batch_response_test = """
{
  "batch_1": {
    "lumayan nyaman,tp kebersihan kmr mandi perlu ditingkatkan lg biar gk ada kuning2 di sudutnya lbh bgs": {
      "kebersihan": 0
    },
    "kamarnya kurang maintenance. paling baik kamar mandi dibersihkan lebih sering. kamar mandi adalah bagian yang penting untuk pelanggan. kebersihan selimut bed sheet dan handuk juga perlu diperhatikan lebih jauh lagi.": {
      "kebersihan": 0,
      "linen": 0
    },
    "harga terjangkau, kamar luas, ada sofa, semua oke, kecuali kamar mandi. seharian keluar bau tak sedap entah apa masalahnya.": {
      "bau": 0
    },
    "Jaringan WiFi dalam kamar lelet": {
      "wifi": 0
    },
    "secara umum bagus, cuma sprei, sarung bantal dan handuk tidak layak...maaf..": {
      "linen": 0
    },
    "Kamar bersih dan nyaman tetapi amenties dari airy tidak disediakan. Begitu juga dengan compliment free snack dari airy": {
      "kebersihan": 1
    },
    "suasana .. and view-nya..kerennnnn... pegawainya ramahh... datang langsung check in... dan diantar ke kamar... untuk fasilitas buat anak2 lumayan... ada perosotan ayunan dan mainan lainnya.. kolam renang juga.. tp kolamnya Dingin... dan kurang bersih.. mohon ditingkatkan jg utk kebersihan kamar nya. thanks airy .. thank ariandri...sukses selalu..": {
      "kebersihan": 0,
      "service": 1
    },
    "WiFi nya tolong diperkuat sinyal dan ditambah bandwithnya": {
      "wifi": 0
    },
    "plus: lokasi tenang, strategis, rate ekonomis. minus: kebersihan kurang, kamar bau, kran bulukan.": {
      "kebersihan": 0,
      "bau": 0
    },
    "kamar luas dan rapi, tapi sayang untuk tv kebetulan sedang jelek sinyalnya dan minta ganti ke kamar yg lain yg memiliki sinyal bagus tidak di perbolehkan karena pesan dari airy rooms dan tidak tersedia wifi": {
      "tv": 0,
      "wifi": 0
    }
  }
}
"""
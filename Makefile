setup:
	pip install -r requirements.txt

train-traditional-ml:
	python src\models\traditional_pipeline\training.py

inference_test_data:
	python src\models\traditional_pipeline\predict.py

inference_llm:
	python src\models\llm_api\parsing.py
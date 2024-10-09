augmentation:
	cd SAM40 && python3 data_augmentation.py

run:
	python3 __main__.py >results.txt

install:
	pip install -r requirements.txt
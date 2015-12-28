# directories
DIR_RAW_DATA := raw_data
DIR_BUILD := build
DIR_FEATURE := $(DIR_BUILD)/feature# features after feature-engineering
DIR_VAL := $(DIR_BUILD)/val# training predictions
DIR_TEST_RESULT := $(DIR_BUILD)/test# test predictions

RAW_DATA_TRAIN := $(DIR_RAW_DATA)/train.csv
RAW_DATA_TEST := $(DIR_RAW_DATA)/test.csv

TARGET_TRAIN := $(DIR_RAW_DATA)/train.y

$(TARGET_TRAIN): $(RAW_DATA_TRAIN)	
	python ./src/generate_train_y.py

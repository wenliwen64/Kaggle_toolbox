# directories
DIR_RAW_DATA := raw_data
DIR_BUILD := build
DIR_FEATURE := $(DIR_BUILD)/feature# features after feature-engineering
DIR_VAL := $(DIR_BUILD)/val# training predictions
DIR_TEST := $(DIR_BUILD)/test# test predictions

TRAIN_RAW_DATA := $(DIR_RAW_DATA)/train.csv
TEST_RAW_DATA := $(DIR_RAW_DATA)/test.csv

TRAIN_TARGET := $(DIR_RAW_DATA)/train.y
TRAIN_ID := $(DIR_RAW_DATA)/train.id
TEST_ID := $(DIR_RAW_DATA)/test.id

pre1: $(TRAIN_TARGET) $(TRAIN_ID) $(TEST_ID)
$(TRAIN_TARGET) $(TRAIN_ID) $(TEST_ID): $(TRAIN_RAW_DATA) $(TEST_RAW_DATA) | src/generate_train_y_and_id.py
	echo $@
	python ./src/generate_train_y_and_id.py --train-raw-file $< --test-raw-file $(word 2, $^) --train-y-file $(TRAIN_TARGET) --train-id-file $(TRAIN_ID) --test-id-file $(TEST_ID)

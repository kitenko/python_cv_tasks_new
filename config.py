import os

DATASET_PATH = 'dataset_image'
LOGS_DIR = 'logs_dir'
DATASET_PATH_RED = os.path.join(DATASET_PATH, 'red_image')
DATASET_PATH_BLUE = os.path.join(DATASET_PATH, 'blue_image')
DATASET_PATH_GREEN = os.path.join(DATASET_PATH, 'green_image')
DATASET_PATH_GRAY = os.path.join(DATASET_PATH, 'gray_image')
DATASET_PATH_COLOUR = os.path.join(DATASET_PATH, 'colour_image')
JSON_FILE_PATH = os.path.join(DATASET_PATH, 'data.json')
CLASS_NAMES = ('red', 'green', 'blue', 'gray', 'colour')
INPUT_SHAPE = (224, 224, 3)
NUMBER_OF_CLASSES = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20

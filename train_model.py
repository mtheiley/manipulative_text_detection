import os
import pandas
import tensorflow
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import Dense, Embedding
from matplotlib import pyplot as plt
from data_preprocessor import DataPreprocessor

# Train Parameters
DATA_BASE_PATH = "./Data/Output/"
DATA_FILE_NAME = "output.csv"
OUTPUT_BASE_PATH = "./Models/"
MODEL_NAME = "model.h5" 
MAX_VOCABULARY_SIZE = 200000
MAX_SENTENCE_LENGTH = 1500
BATCH_SIZE = 15
PREFETCH_SIZE = 5
TRAIN_DATASET_RATIO = 0.7
VALIDATE_DATASET_RATIO = 0.2
TEST_DATASET_RATIO = 0.1
EMBEDDING_SIZE = 32
OUTPUT_TENSOR_SIZE = 3
LOSS_TYPE = "BinaryCrossentropy"
OPTIMIZER_TYPE = "Adam"
NUMBER_OF_EPOCHS = 1 
SAVE_FORMAT="h5"

# Perform Data Preprocessing
processor = DataPreprocessor()
processor.setMaxVocabularySize(MAX_VOCABULARY_SIZE)
processor.setMaxSentenceLength(MAX_SENTENCE_LENGTH)

dataPath = os.path.join(DATA_BASE_PATH, DATA_FILE_NAME)
processor.process(dataPath)

processor.setBatchSize(BATCH_SIZE)
processor.setPrefetchSize(PREFETCH_SIZE)

processor.setTrainRatio(TRAIN_DATASET_RATIO)
processor.setValidateRatio(VALIDATE_DATASET_RATIO)
processor.setTestRatio(TEST_DATASET_RATIO)

processor.formDataset()

# Build Neural Net Model
model = Sequential()
model.add(Embedding(MAX_VOCABULARY_SIZE + 1, EMBEDDING_SIZE))
model.add(Bidirectional(LSTM(EMBEDDING_SIZE, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(OUTPUT_TENSOR_SIZE, activation='sigmoid'))

model.compile(loss=LOSS_TYPE, optimizer=OPTIMIZER_TYPE)
model.summary()

# Train
trainRun = model.fit(processor.getTrainSet(), epochs=NUMBER_OF_EPOCHS, validation_data=processor.getValidateSet())
print(trainRun.history)

# Save Model
modelPath = os.path.join(OUTPUT_BASE_PATH, MODEL_NAME)
model.save(modelPath, save_format=SAVE_FORMAT)
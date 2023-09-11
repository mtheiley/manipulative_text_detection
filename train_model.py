import os
import pandas
import tensorflow
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional, Dense, Embedding
from matplotlib import pyplot as plt

# Parameters
DATA_BASE_PATH = "./Data/Output/"
DATA_FILE_NAME = "output.csv"
OUTPUT_BASE_PATH = "./Models/"
MODEL_NAME = "model.keras" 
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

# Load Data
dataPath = os.path.join(DATA_BASE_PATH, DATA_FILE_NAME)
dataFrame = pandas.read_csv(dataPath)

categories = dataFrame.columns[2:]
commentList = dataFrame['comment'].values
labelList = dataFrame[categories].values

commentList = commentList.astype(str) #Ensure type is string
labelList = labelList.astype(int) #Ensure type is int

sampleSize = len(commentList)

# Vectorize Data
VECTORIZATION_MODE = 'int'

textVectorizer = TextVectorization(
    max_tokens=MAX_VOCABULARY_SIZE,
    output_sequence_length=MAX_SENTENCE_LENGTH,
    output_mode=VECTORIZATION_MODE)

textVectorizer.adapt(commentList)
textVectorList = textVectorizer(commentList)

# Form The Dataset
dataset = tensorflow.data.Dataset.from_tensor_slices((textVectorList, labelList))
dataset = dataset.cache()
dataset = dataset.shuffle(sampleSize)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(PREFETCH_SIZE)

datasetSize = len(dataset)
trainSetSize = int(datasetSize * TRAIN_DATASET_RATIO)
validateSetSize = int(datasetSize * VALIDATE_DATASET_RATIO)
testSetSize = int(datasetSize * TRAIN_DATASET_RATIO)

trainSet = dataset.take(trainSetSize)
validateSet = dataset.skip(trainSetSize).take(validateSetSize)
testSet = dataset.skip(trainSetSize + validateSetSize).take(testSetSize)

# Setup Neural Net Model

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
#trainRun = model.fit(trainSet, epochs=NUMBER_OF_EPOCHS, validation_data=validateSet)
#print(trainRun.history)

# Save Model
#modelPath = os.path.join(OUTPUT_BASE_PATH, MODEL_NAME)
#model.save()
import os
import tensorflow
import keras
import numpy
from data_preprocessor import DataPreprocessor

DATA_BASE_PATH = "./Data/Output/"
DATA_FILE_NAME = "output.csv"
MODEL_NAME = "model.h5"
MODEL_BASE_PATH = "./Models/"
MAX_VOCABULARY_SIZE = 200000
MAX_SENTENCE_LENGTH = 1500

model = keras.models.load_model(os.path.join(MODEL_BASE_PATH, MODEL_NAME))

processor = DataPreprocessor()
processor.setMaxVocabularySize(MAX_VOCABULARY_SIZE)
processor.setMaxSentenceLength(MAX_SENTENCE_LENGTH)

dataPath = os.path.join(DATA_BASE_PATH, DATA_FILE_NAME)
processor.process(dataPath)

while(True):
    inputText = input("Write text to analyse: ")
    vectorizedText = processor.vectorizeText(inputText)
    result = model.predict(numpy.array([vectorizedText]))
    print(result)

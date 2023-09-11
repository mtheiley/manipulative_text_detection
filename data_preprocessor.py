import os
import pandas
import tensorflow
import numpy
from tensorflow.keras.layers import TextVectorization

class DataPreprocessor:
    def __init__(self):
        self.maxVocabularySize = 0
        self.maxSentenceLength = 0
        self.vectorizationMode = 'int'
            
        self.trainRatio = 0
        self.validateRatio = 0
        self.testRatio = 0
        self.batchSize = 0
        self.prefetchSize = 0

        self.dataFrame = None
        self.categories = None
        self.commentList = None
        self.labelList = None
        self.sampleSize = 0
        self.textVectorizer = None
        self.textVectorList = None

        self.dataset = None
        self.datasetSize = None
        self.trainSetSize = None
        self.validateSetSize = None
        self.testSetSize = None

        self.trainSet = None
        self.validateSet = None
        self.testSet = None

    def getTrainSet(self):
        return self.trainSet

    def getValidateSet(self):
        return self.validateSet

    def getTestSet(self):
        return self.testSet

    def getVectorizedList(self):
        return self.textVectorList

    def getCommentList(self):
        return self.commentList

    def getLabelList(self):
        return self.labelList

    def getSampleSize(self):
        return self.sampleSize

    def setMaxVocabularySize(self, maxVocabularySize):
        self.maxVocabularySize = maxVocabularySize
    
    def setMaxSentenceLength(self, maxSentenceLength):
        self.maxSentenceLength = maxSentenceLength

    def setTrainRatio(self, trainRatio):
        self.trainRatio = trainRatio

    def setValidateRatio(self, validateRatio):
        self.validateRatio = validateRatio

    def setTestRatio(self, testRatio):
        self.testRatio = testRatio

    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
    
    def setPrefetchSize(self, prefetchSize):
        self.prefetchSize = prefetchSize

    def vectorizeText(self, text):
        return self.textVectorizer(text)

    def __load(self, dataPath):
        self.dataFrame = pandas.read_csv(dataPath)

        self.categories = self.dataFrame.columns[2:]
        self.commentList = self.dataFrame['comment'].values
        self.labelList = self.dataFrame[self.categories].values

        self.commentList = self.commentList.astype(str) #Ensure type is string
        self.labelList = self.labelList.astype(int) #Ensure type is int

        self.sampleSize = len(self.commentList)

    def __vectorize(self):
        self.textVectorizer = TextVectorization(
            max_tokens=self.maxVocabularySize,
            output_sequence_length=self.maxSentenceLength,
            output_mode=self.vectorizationMode)

        self.textVectorizer.adapt(self.commentList)
        self.textVectorList = self.textVectorizer(self.commentList)

    def formDataset(self):
        datasetInputs = (self.getVectorizedList(), self.getLabelList())
        self.dataset = tensorflow.data.Dataset.from_tensor_slices(datasetInputs)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(self.getSampleSize())
        self.dataset = self.dataset.batch(self.batchSize)
        self.dataset = self.dataset.prefetch(self.prefetchSize)

        self.datasetSize = len(self.dataset)
        self.trainSetSize = int(self.datasetSize * self.testRatio)
        self.validateSetSize = int(self.datasetSize * self.validateRatio)
        self.testSetSize = int(self.datasetSize * self.testRatio)

        self.trainSet = self.dataset.take(self.trainSetSize)
        self.validateSet = self.dataset.skip(self.trainSetSize).take(self.validateSetSize)
        self.testSet = self.dataset.skip(self.trainSetSize + self.validateSetSize).take(self.testSetSize)

    def process(self, dataPath):
        self.__load(dataPath)
        self.__vectorize()
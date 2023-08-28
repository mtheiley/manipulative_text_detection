import csv
import os

INPUT_DATA_FOLDER = './Data/Input'
OUTPUT_DATA_FOLDER = './Data/Output'
IN_FILE_1 = 'toxic_comments.csv'
IN_FILE_2 = 'unhealthy_comments.csv'
IN_FILE_3 = 'youtube_toxic_comments.csv'
OUT_FILE = 'output.csv'

class Comment:
    def __init__(self, text):
        self.text = text
        self.isManipulative = False
        self.isToxic = False
        self.isNormal = False

class FileProcessor:

    def setInputDataFolder(self, inputDataFolder):
        self.FP_INPUT_DATA_FOLDER = inputDataFolder

    def setOutputDataFolder(self, outputDataFolder):
        self.FP_OUTPUT_DATA_FOLDER = outputDataFolder

    def __buildTitleSet(self, row):
        self.titles = row
        for i in range(len(self.titles)):
            self.columnTitleLookup[self.titles[i]] = i
        
        self.textColumnNameIndex = self.columnTitleLookup[self.textColumnName]

    def __rowMatches(self, row, conditionSet, conditionEnabled):
        if conditionEnabled == False:
            return False
        
        for conditionName in conditionSet:
            index = self.columnTitleLookup[conditionName]
            expectedValue = conditionSet[conditionName]
            actualValue = row[index]
            if expectedValue == actualValue:
                return True
        return False

    def __getCommentText(self, row):
        return row[self.textColumnNameIndex]

    def __isManipulative(self, row):
        return self.__rowMatches(row, self.conditionsManipulative, self.shouldBuildManipulative)

    def __isToxic(self, row):
        return self.__rowMatches(row, self.conditionsToxic, self.shouldBuildToxic)

    def __isNormal(self, row):
        return self.__rowMatches(row, self.conditionsNormal, self.shouldBuildNormal)

    def __cleanString(self, string):
        stringEncoded = string.encode("ascii", "ignore")
        return stringEncoded.decode()

    def __tryAddComment(self, comment):
        if comment.text in self.comments:
            self.duplicateComments += 1
            return False
        self.comments[comment.text] = comment

        if(comment.isManipulative):
            self.addedManipulativeComments += 1
        if(comment.isToxic):
            self.addedToxicComments += 1
        if(comment.isNormal):
            self.addedNormalComments += 1
        self.addedComments += 1

    def __init__(self):
        self.titles = []
        self.shouldBuildManipulative = False
        self.conditionsManipulative = {}
        self.shouldBuildToxic = False
        self.conditionsToxic = {}
        self.shouldBuildNormal = False
        self.conditionsNormal = {}
        self.columnTitleLookup = {}
        self.comments = {}
        self.textColumnName = ''
        self.textColumnNameIndex = 0
        self.addedComments = 0
        self.addedManipulativeComments = 0
        self.addedToxicComments = 0
        self.addedNormalComments = 0
        self.duplicateComments = 0
        self.FP_OUTPUT_FILE_HEADER = ['id', 'comment', 'manipulative', 'toxic', 'normal']
        self.FP_INPUT_DATA_FOLDER = ''
        self.FP_OUTPUT_DATA_FOLDER = ''

    def setTextColumnName(self, textColumnName):
        self.textColumnName = textColumnName

    def buildManipulativeComments(self, conditions):
        self.conditionsManipulative = conditions
        self.shouldBuildManipulative = True

    def skipManipulativeComments(self):
        self.shouldBuildManipulative = False

    def buildToxicComments(self, conditions):
        self.conditionsToxic = conditions
        self.shouldBuildToxic = True

    def skipToxicComments(self):
        self.shouldBuildToxic = False

    def buildNormalComments(self, conditions):
        self.conditionsNormal = conditions
        self.shouldBuildNormal = True

    def skipNormalComments(self):
        self.shouldBuildNormal = False

    def processFile(self, fileName):
        if(self.setTextColumnName == ''):
            print("Error no text column specified!")
            return
        if(self.shouldBuildManipulative == False and self.shouldBuildToxic == False and self.shouldBuildNormal == False):
            print("No builders specified!")
            return

        self.addedComments = 0
        self.addedManipulativeComments = 0
        self.addedToxicComments = 0
        self.addedNormalComments = 0
        self.duplicateComments = 0

        with open(os.path.join(self.FP_INPUT_DATA_FOLDER, fileName)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    self.__buildTitleSet(row)
                    line_count += 1
                else:
                    comment = Comment(self.__getCommentText(row))
                    shouldAddComment = False
                    if(self.__isManipulative(row)):
                        comment.isManipulative = True
                        shouldAddComment = True
                    if(self.__isToxic(row)):
                        comment.isToxic = True
                        shouldAddComment = True
                    if(self.__isNormal(row)):
                        comment.isNormal = True
                        shouldAddComment = True
                    if(shouldAddComment):
                        self.__tryAddComment(comment)
                    
                    line_count += 1
            print(f'File \'{fileName}\'processing done! read {line_count} lines. Added {self.addedComments} comments. Skipping {self.duplicateComments} duplicates...')
            print(f'{self.addedManipulativeComments} manipulative, {self.addedToxicComments} toxic. {self.addedNormalComments} normal.')

    def writeCSVFile(self, fileName):
        with open(os.path.join(self.FP_OUTPUT_DATA_FOLDER, fileName), 'w', encoding='UTF8') as FILE:
            writer = csv.writer(FILE)
            writer.writerow(self.FP_OUTPUT_FILE_HEADER)
            
            commentId = 0
            for comment in self.comments.values():
                isManipulative = str(int(comment.isManipulative))
                isToxic = str(int(comment.isToxic))
                isNormal = str(int(comment.isNormal))
                cleanedText = self.__cleanString(comment.text)
                data = [str(commentId), cleanedText, isManipulative, isToxic, isNormal]
                writer.writerow(data)
                commentId += 1

## Setup the File Processor

fileProcessor = FileProcessor()
fileProcessor.setInputDataFolder(INPUT_DATA_FOLDER)
fileProcessor.setOutputDataFolder(OUTPUT_DATA_FOLDER)

## Process File 1 - Toxic Comments

fileProcessor.setTextColumnName('comment_text')
fileProcessor.buildManipulativeComments({
    'threat' : '1'
})
fileProcessor.buildToxicComments({
    'toxic' : '1',
})
fileProcessor.buildNormalComments({
    'toxic' : '0'
})

fileProcessor.processFile(IN_FILE_1)

## Process File 2 - Unhealthy Comments

fileProcessor.setTextColumnName('comment')
fileProcessor.buildManipulativeComments({
    'antagonize' : '1',
    'dismissive' : '1'
})
fileProcessor.buildToxicComments({
    'antagonize' : '1',
    'dismissive' : '1',
    'hostile' : '1'
})
fileProcessor.skipNormalComments()

fileProcessor.processFile(IN_FILE_2)

## Process File 3 - Toxic Youtube Comments

fileProcessor.setTextColumnName('Text')
fileProcessor.buildManipulativeComments({
    'IsThreat' : 'TRUE',
    'IsProvocative' : 'TRUE'
})
fileProcessor.buildToxicComments({
    'IsToxic' : 'TRUE'
})
fileProcessor.buildNormalComments({
    'IsToxic' : 'FALSE'
})

fileProcessor.processFile(IN_FILE_3)

# Create the output CSV file

fileProcessor.writeCSVFile(OUT_FILE)
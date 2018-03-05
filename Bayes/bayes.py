'''
Created on Mar 3, 2018
bayes:

Input:

Output:
@author: ZhengjunZhao
'''
from numpy import *

class BayesOperator():
    def __init__(self):
        pass


    def loadDataSet(self):
        '''
        创建实验样本
        :return:postingList,classVec
        postingList:词条集合
        classVec:类别标签集合,0和1
        '''
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
        return postingList, classVec

    def createVocabList(self,dataSet):
        '''
        创建一个包含在所有文档中出现不重复的列表
        :param dataSet:
        :return:
        '''
        vocabSet = set([])  # create empty set
        for document in dataSet:
            # print(document)
            vocabSet = vocabSet | set(document)  # union of the two sets
        return list(vocabSet)


    def setOfWords2Vec(self,vocabList, inputSet):
        '''
        生成词向量
        :param vocabList:词汇表
        :param inputSet:文档
        :return:文档向量
        '''
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print("the word: %s is not in my Vocabulary!" % word)
        return returnVec

    def trainNB0(self,trainMatrix,trainCategory):
        '''
        朴素贝叶斯分类器训练函数
        :param trainMatrix:文档矩阵
        :param trainCategory:对应的文档类别标签
        :return:
        '''
        numTrainDocs = len(trainMatrix)                     # 文档数量
        print('numTrainDocs = ', numTrainDocs)
        numWords = len(trainMatrix[0])                      # 第一个文档词汇数量
        print('numWords = ', numWords)
        print('sum(trainCategory) = ', sum(trainCategory))
        print('float(numTrainDocs) = ', float(numTrainDocs))
        pAbusive = sum(trainCategory) / float(numTrainDocs) # 总类别数量/文档长度
        p0Num = zeros(numWords)
        p1Num = zeros(numWords)
        p0Denom = 0.0
        p1Denom = 0.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        p1Vect = p1Num / p1Denom        # change to log()
        p0Vect = p0Num / p0Denom
        return p0Vect,p1Vect,pAbusive

if __name__ == '__main__':
    bayes=BayesOperator()
    listOPosts, listClasses = bayes.loadDataSet()
    print(len(listOPosts[0]))
    myVocabList = bayes.createVocabList(listOPosts)
    # print(myVocabList)
    # print(bayes.setOfWords2Vec(myVocabList,listOPosts[0]))

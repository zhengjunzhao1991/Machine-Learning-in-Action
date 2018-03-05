'''
Created on Mar 3, 2018
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: ZhengjunZhao
'''

from numpy import *
import operator
import matplotlib.pyplot as plt

class K_nn():
    def __init__(self):
        pass

    def createDataSet(self):
        group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
        labels = ['A','A','B','B']
        return group,labels

    def classify0(self,inX,dataSet,labels,k):
        '''
        分类
        :param inX: 用于分类的输入向量,即将对其进行分类
        :param dataSet: 输入的训练样本集
        :param labels: 标签向量
        :param k: 最近邻居的数目
        ps:标签向量的元素数目和矩阵dataSet的行数相同
        :return:
        '''
        dataSetSize = dataSet.shape[0]      #得到数组的行数.即知道有几个训练数据

        # 距离计算
        # tile:Construct an array by repeating A the number of times given by reps
        # diffMat:得到目标与训练数值之间的差值
        diffMat = tile(inX, (dataSetSize,1)) - dataSet
        # 各个元素分别平方
        sqDiffMat = diffMat**2
        # 对应列相加，即得到了每一个距离的平方
        sqDistances = sqDiffMat.sum(axis=1)
        # 开方，得到距离
        distances = sqDistances**0.5
        # 升序排列
        # argsort函数返回的是数组值从小到大的索引值
        sortedDistIndicies = distances.argsort()
        # print('sortedDistIndicies = ',sortedDistIndicies)
        # 选择距离最小的k个点
        classCount = {}
        for i in range(k):
            # print('i = ',i)
            # print('sortedDistIndicies[i] = ', sortedDistIndicies[i])
            voteIlabel = labels[sortedDistIndicies[i]]      # 对应标签
            # get是取字典里的元素，如果之前这个voteIlabel是有的，
            # 那么就返回字典里这个voteIlabel里的值，
            # 如果没有就返回0（后面写的），这行代码的意思就是
            # 算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
            classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        # key=operator.itemgetter(1)的意思是按照字典里的第一个排序，
        # {A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
        sortedClassCount = sorted(classCount.items(),
                                  key=lambda item:item[1],reverse=True)
        return sortedClassCount[0][0]

    def file2matrix(self,filename):
        '''
        讲文本记录转换为Numpy的解析程序
        :param filename:
        :return:
        '''
        fr = open(filename)     # 打开文件
        arrayOLines = fr.readlines()        # 读一次性读取文件的每一行
        numberOfLines = len(arrayOLines)    # 总行数
        returnMat = zeros((numberOfLines,3))    # 创建相应矩阵并置零
        classLabelVector=[]                 # 对应标签
        index = 0                           # 行标
        for line in arrayOLines:
            line = line.strip()             # 去首尾空格
            listFromLine = line.split('\t') # 以制表符分割数据
            returnMat[index,:] = listFromLine[0:3]  #将对应的特征数据存入矩阵
            classLabelVector.append(int(listFromLine[-1]))  #存入标签
            index+=1
        return returnMat,classLabelVector

    def firstPlot(self,filename):
        filename = './datingTestSet2.txt'
        datingDataMat, datingLabels = self.file2matrix(filename)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
        ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
                   15.0 * array(datingLabels), 15.0 * array(datingLabels))
        plt.show()

    def autoNorm(self,dataSet):
        '''
        归一化
        :param dataSet: 数据集
        :return:
        '''
        minVals = dataSet.min(0)    # 每列最小值
        maxVals = dataSet.max(0)    # 每列最大值
        ranges = maxVals - minVals  # 取值范围
        normDataSet = zeros(shape(dataSet))     # 创建同等大小零矩阵
        m = dataSet.shape[0]        # 总行数
        normDataSet = dataSet - tile(minVals,(m,1))     # 减去最小值
        normDataSet = normDataSet / tile(ranges,(m,1))  # 除以范围,归一化
        return normDataSet,ranges,minVals

    def datingClassTest(self):
        '''
        预测测试函数
        :return:
        '''
        hoRatio = 0.10  # hold out 10%
        datingDataMat, datingLabels = self.file2matrix('datingTestSet2.txt')  # load data setfrom file
        normMat, ranges, minVals = self.autoNorm(datingDataMat)         # 归一化
        m = normMat.shape[0]                        # 总行数
        numTestVecs = int(m * hoRatio)              # 样本集起始行数
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = self.classify0(normMat[i, :],        # 测试集
                                              normMat[numTestVecs:m, :],        # 样本集
                                              datingLabels[numTestVecs:m],      # 对应标签
                                              3)
            print("the classifier came back with: %d, the real answer is: %d"
                  % (classifierResult, datingLabels[i]))
            if (classifierResult != datingLabels[i]):
                errorCount += 1.0
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
        print(errorCount)

    def classifyPerson(self):
        '''
        约会网站预测函数
        :return:
        '''
        resultList = ['not at all','in small doses','in large doses']
        ffMiles = float(input("frequent flier miles earned per year?"))
        percentTats = float(input("percentage of time spent playing video game?"))
        iceCream = float(input("liters of ice cream consumed per year?"))
        datingDataMat, datingLabels = self.file2matrix('datingTestSet2.txt')  # load data setfrom file
        normMat, ranges, minVals = self.autoNorm(datingDataMat)         # 归一化
        inArr = array([ffMiles,percentTats,iceCream])
        classifierResult = self.classify0((inArr-minVals)/ranges,  # 测试集
                                          normMat,  # 样本集
                                          datingLabels,  # 对应标签
                                          3)
        print('You will probably like this person: ',resultList[classifierResult - 1])

    def img2vector(self,filename):
        '''
        将图片转换为向量
        :return:
        '''
        returnVect = zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect

if __name__ == '__main__':
    k_nn = K_nn()
    group, labels = k_nn.createDataSet()
    # print(k_nn.classify0([0,0],group,labels,3))
    filename = './datingTestSet2.txt'
    datingDataMat, datingLabels=k_nn.file2matrix(filename)

    # k_nn.firstPlot(filename)

    normDataSet, ranges, minVals = k_nn.autoNorm(datingDataMat)
    # print(normDataSet)
    # k_nn.datingClassTest()
    k_nn.classifyPerson()
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

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
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

if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0,0],group,labels,3))

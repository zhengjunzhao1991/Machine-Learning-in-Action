from kNN import K_nn as k_nn
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

knn = k_nn()
# group,labels = knn.createDataSet()
# print(knn.classify0([0,0],group,labels,3))

filename = './datingTestSet2.txt'
datingDataMat,datingLabels=knn.file2matrix(filename)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
           15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# datingDataMat,datingLabels = knn.file2matrix('datingTestSet2.txt')
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
# ax.axis([-2,25,-0.2,2.0])
# plt.xlabel('Percentage of Time Spent Playing Video Games')
# plt.ylabel('Liters of Ice Cream Consumed Per Week')
# plt.show()
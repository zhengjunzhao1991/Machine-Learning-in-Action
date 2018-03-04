import kNN as k_nn
knn = k_nn()
group,labels = knn.createDataSet()
print(knn.classify0([0,0],group,labels,3))
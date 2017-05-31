import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import  KNeighborsClassifier
import collections

class NearestNeighborClassification(object):
  def __init__(self, batch_size=32, k=3):
    self.batch_size = batch_size
    self.k = k

  def fit(self, x, y):
    self.x, self.y = x, y

  def predict(self, x):
  	'''using l2 norm'''
  	y_ = np.empty((x.shape[0],),dtype=self.y.dtype) #DU HAR JO LAGRET DET SOM STREEENGERRRRRR
  	index=0
  	print x.shape
  	print self.x.shape
  	for element in x:
  		squared_l2_dist_to_element = np.sum(np.square(self.x-element),axis=1)
  		sorted_classes_of_elements = self.y[squared_l2_dist_to_element.argsort()]
  		most_common_neighbour = (collections.Counter(sorted_classes_of_elements[0:self.k]).most_common()[0])[0]
  		#print most_common_neighbour
  		y_[index] = most_common_neighbour
  		index+=1
  	return y_


  def get_params(self, deep=True):
    """
    Just here to follow sklearn standard, so it can be used along with library functions
    :param deep:
    :return : empty dict
    """
    return {}

  def set_params(self, **prameters):
    """
    Just here to follow sklearn standard, so it can be used along with library functions
    :param prameters:
    :return: None
    """
    pass


def evaluate_knn(k, n_samples=100, n_features=300, n_informative=5):
  from sklearn import datasets
  X_1, y_1 = datasets.make_classification(n_samples=n_samples,
                                          n_features=n_features, n_informative=n_informative,
                                          n_redundant=2 if (n_features-n_informative)>2 else
                                          0,
                                          random_state=1)
  accuracy = 0
  return accuracy


def evaluate_knn_crossval(k, n_samples=100, n_features=300, n_informative=5):
  from sklearn import datasets
  X_1, y_1 = datasets.make_classification(n_samples=n_samples,
                                          n_features=n_features, n_informative=n_informative,
                                          n_redundant=2 if (n_features-n_informative)>2 else
                                          0,
                                          random_state=1)
  accuracy = 0
  return accuracy


def plot_classification(data, labels, clf):
  colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

  # create a mesh to plot in
  x_range, y_range = np.abs(data[:, 0].max() - data[:, 0].min()), np.abs(data[:, 1].max() - data[:, 1].min())
  space = .5
  x_min, x_max = data[:, 0].min() - x_range*space, data[:, 0].max() + x_range*space
  y_min, y_max = data[:, 1].min() - x_range*space, data[:, 1].max() + y_range*space

  xx, yy = np.meshgrid(np.linspace(x_min, x_max),
                       np.linspace(y_min, y_max))
  Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
  for i, c in enumerate(np.unique(labels)):
    c_indices = labels == c
    plt.scatter(data[c_indices, 0], data[c_indices, 1], c=colors[i%len(colors)])


def main():
  from sklearn.model_selection import cross_val_predict
  from sklearn.datasets import load_breast_cancer

  clf = NearestNeighborClassification()
  data = load_breast_cancer()
  x, y = data.data. data.target
  print (cross_val_predict(clf, x, y)==y).mean()



if __name__ == '__main__':
  main()
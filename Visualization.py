from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt



class DecisionBoundary(object):


    def __init__(self,X,y,resolution=0.02):
        self.X = X
        self.y = y
        self.markers = ('s', 'x', 'o', '^', 'v')
        self.colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        self.cmap = ListedColormap(self.colors[:len(np.unique(y))])
        self.x1_min, self.x1_max = X[:, 0].min() -1 , X[:, 0].max() +1
        self.x2_min, self.x2_max = X[:, 1].min() -1 , X[:, 1].max() + 1

        self.xx1, self.xx2 = np.meshgrid(np.arange(self.x1_min, self.x1_max, resolution),
                               np.arange(self.x2_min, self.x2_max, resolution))


    def plot_decision_boundary(self,classifier,itr):
        Z = classifier.predict(np.array([self.xx1.ravel(), self.xx2.ravel()]).T)
        Z = Z.reshape(self.xx1.shape)

        fig = plt.figure()

        cont = plt.contourf(self.xx1, self.xx2, Z, alpha=0.3, cmap=self.cmap)
        plt.xlim(self.xx1.min(), self.xx1.max())
        plt.ylim(self.xx2.min(), self.xx2.max())

        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X[self.y == cl, 0],
                        y=self.X[self.y == cl, 1],
                        c=self.colors[idx],
                        alpha=0.8,
                        marker=self.markers[idx],
                        label=cl,
                        edgecolors='black')
        plt.savefig('img'+str(itr))
        plt.clf()



from Visualization import DecisionBoundary
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):


    def __init__(self,learning_rate=0.01, n_iter=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')


    def fit(self,X,y):
        self.db = DecisionBoundary(X,y)
        random_generator = np.random.RandomState(self.random_state)
        self.weights = random_generator.normal(loc=0.0, scale=0.01, size= X.shape[1])
        self.bias = random_generator.randint(1,1000)
        self.errors_ = []


        for i in range(self.n_iter):
            self.iteration = i

            errors = 0
            for xi, target in zip(X,y):
                error = target - self.predict(xi)
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += int(self.learning_rate * error != 0.0)

            self.db.plot_decision_boundary(self,self.iteration)
            self.errors_.append(errors)

        return self

    def activation_function(self,X):
        return np.dot(X,self.weights) + self.bias


    def predict(self, X):
        return np.where(self.activation_function(X) >= 0.0, 1, -1)


import pandas as pd
from Perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from Visualization import DecisionBoundary
df = pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)

X= df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa',-1,1)


per = Perceptron(learning_rate=0.1,n_iter=100)
per.fit(X,y)
plt.plot(range(1,len(per.errors_)+1), per.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


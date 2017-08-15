import numpy as np
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

class Neural_Network(object):
    def __init__(self, inputLayerSize=2, outputLayerSize=1, hiddenLayerSize=5, func='tanh', learning='momentum'):
        np.random.seed(0)
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize

        self.W1 = np.random.randn(self.inputLayerSize + 1, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize + 1, self.outputLayerSize)

        self.gradStep = 0.0001
        self.reg_lambda = 0.005

        self.func = func
        self.learning = learning

        self.vW1 = 0.0
        self.vW2 = 0.0
        self.vb1 = 0.0
        self.vb2 = 0.0

        self.mu = 0.99

    def forward(self, X):
        self.z2 = np.dot(X, self.W1[:-1,:]) + self.W1[-1,:]
        self.a2 = self.transfer(self.z2)
        self.z3 = np.dot(self.a2, self.W2[:-1,:]) + self.W2[-1,:]
        yHat = self.transfer(self.z3)
        return yHat

    def costFunction(self, X, y):
        self.yHat = self.forward(X)

        cost = 0.5 * np.sum((y - self.yHat)**2)
        
        return cost

    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.transfer(self.z3, deriv=True))
        dJdW2 = np.dot(self.a2.T, delta3)
        dJdb2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.W2[:-1,:].T)*self.transfer(self.z2, deriv=True)
        dJdW1 = np.dot(X.T, delta2)
        dJdb1 = np.sum(delta2, axis=0, keepdims=True)

        dJdW1 = dJdW1 + self.W1[:-1,:] * self.reg_lambda
        dJdW2 = dJdW2 + self.W2[:-1,:] * self.reg_lambda

        return dJdW1, dJdW2, dJdb1, dJdb2

    def backward(self, X, y):
        dJdW1, dJdW2, dJdb1, dJdb2 = self.costFunctionPrime(X,y)

        if self.learning == 'momentum':
            self.vW1 = self.mu * self.vW1 - self.gradStep * dJdW1
            self.vb1 = self.mu * self.vb1 - self.gradStep * dJdb1
            self.vW2 = self.mu * self.vW2 - self.gradStep * dJdW2
            self.vb2 = self.mu * self.vb2 - self.gradStep * dJdb2

        elif self.learning == 'NAG':
            self.vW1 = -(self.mu * self.vW1) + (1. + self.mu) * (self.mu * self.vW1 - self.gradStep * dJdW1)
            self.vb1 = -(self.mu * self.vb1) + (1. + self.mu) * (self.mu * self.vb1 - self.gradStep * dJdb1)
            self.vW2 = -(self.mu * self.vW2) + (1. + self.mu) * (self.mu * self.vW2 - self.gradStep * dJdW2)
            self.vb2 = -(self.mu * self.vb2) + (1. + self.mu) * (self.mu * self.vb2 - self.gradStep * dJdb2)

        elif self.learning == 'standard':
            self.vW1 = - self.gradStep * dJdW1
            self.vb1 = - self.gradStep * dJdb1
            self.vW2 = - self.gradStep * dJdW2
            self.vb2 = - self.gradStep * dJdb2

        self.W1[:-1,:] = self.W1[:-1,:] + self.vW1
        self.W1[-1,:] = self.W1[-1,:] + self.vb1

        self.W2[:-1,:] = self.W2[:-1,:] + self.vW2
        self.W2[-1,:] = self.W2[-1,:] + self.vb2

    def train(self, X, y, iterate=1000):
        print 'Initial cost: ', self.costFunction(X,y)
        for i in xrange(iterate):
            self.backward(X,y)
            if int(i+1) % (iterate/10) == 0: print i+1, self.costFunction(X,y)

    def transfer(self, z, deriv=False):
        if self.func=='sigmoid':
            if (deriv==False):
                return 1./(1.+np.exp(-z))
            else:
                return np.exp(-z)/((1.+np.exp(-z))**2)
        elif self.func=='tanh':
            if (deriv==False):
                return np.tanh(z)
            else:
                return (1./np.cosh(z))**2


def normalise(X,y):
    X = (X - np.amin(X, axis=0))/(np.amax(X, axis=0) - np.amin(X, axis=0))
    y = (y - np.amin(y, axis=0))/(np.amax(y, axis=0) - np.amin(y, axis=0))

    return X,y

def standardise(d, mean=None, std=None):
    if (mean is None and std is None):
        mean = np.mean(d, axis=0)
        std = np.std(d, axis=0)
    return (d - mean) / std, mean, std

def destandardise(d, mean, std):
    return (d * std) + mean

test = int(raw_input('Input test number: '))

inputLayerSize = 2

if test == 1:
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

elif test == 2:
    X, y = sklearn.datasets.make_moons(200, noise=0.1)
    y = y.reshape(y.size,1).astype(float)

elif test == 3:
    X, y = sklearn.datasets.make_circles(200, noise=0.1, factor = 0.1)
    y = y.reshape(y.size,1).astype(float)

elif test == 4:
    X = np.array(([0,0], [0,1], [1,0], [1,1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

elif test == 5:
    X = np.linspace(-5*np.pi, 5*np.pi, 201)
    X = X.reshape(X.size,1).astype(float)
    y = np.sinc(X)
    inputLayerSize = 1

X, meanX, stdX = standardise(X)
y, meany, stdy = standardise(y)
NN = Neural_Network(inputLayerSize=inputLayerSize, hiddenLayerSize=10,
                    func='tanh', learning='momentum')
NN.train(X,y,iterate=100000)

if test == 1:
    print destandardise(NN.forward(X), meany, stdy)

elif test == 2 or test == 3:
    X = destandardise(X, meanX, stdX)
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], s=40, c=y, cmap = plt.cm.Spectral, zorder = 1)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    X_in, meanX, stdX = standardise(np.c_[XX.ravel(), YY.ravel()], meanX, stdX)

    Z = NN.forward(X_in)
    Z = destandardise(Z,meany,stdy)
    Z = Z.reshape(XX.shape)

    ax.pcolormesh(XX, YY, Z, cmap=plt.cm.RdBu, alpha = 0.5, zorder = 0)
    plt.show()

elif test == 4:
    print destandardise(NN.forward(X), meany, stdy)

elif test == 5:
    y_predicted = NN.forward(X)
    #y_predicted = destandardise(NN.forward(X), meany, stdy)
    #y = destandardise(y, meany, stdy)
    X = destandardise(X, meanX, stdX)

    plt.plot(X,y)
    plt.plot(X,y_predicted)
    plt.show()




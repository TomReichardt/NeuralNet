import numpy as np
import random as rn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

#np.seterr(all='raise')

class Neural_Network(object):
    def __init__(self, inputLayerSize=2, outputLayerSize=1, hiddenLayerSize=5,
                 batch_size=64, func='tanh', learning='momentum'):
        np.random.seed(0)
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize

        self.W1 = np.random.randn(self.inputLayerSize + 1, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize + 1, self.outputLayerSize)

        self.gradStep = 0.005
        self.reg_lambda = 0.0

        self.func = func
        self.learning = learning

        self.vs = np.zeros(4)

        self.mu = 0.5
        self.batch_size = batch_size
        self.cache = np.zeros(4)
        self.cachev = np.zeros(4)
        self.cachedJ = np.zeros(4)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.decay = 0.9
        self.error_history = np.empty((0,2))
        self.epsilon = 1e-8

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

    def backward(self, X, y, t):

        Xy = zip(X,y)
        rn.shuffle(Xy)
        X_shuff, y_shuff = zip(*Xy)
        X_shuff = np.array(X_shuff)
        y_shuff = np.array(y_shuff)

        for i in xrange(0, X_shuff.shape[0], self.batch_size):

            dJdW1, dJdW2, dJdb1, dJdb2 = self.costFunctionPrime(X_shuff[i:i+self.batch_size],y_shuff[i:i+self.batch_size])

            self.calculate_vs(np.array([dJdW1, dJdb1, dJdW2, dJdb2]), t)

            self.W1[:-1,:] = self.W1[:-1,:] + self.vs[0]
            self.W1[-1,:] = self.W1[-1,:] + self.vs[1]

            self.W2[:-1,:] = self.W2[:-1,:] + self.vs[2]
            self.W2[-1,:] = self.W2[-1,:] + self.vs[3]

    def calculate_vs(self, dJs, t):
        if self.learning == 'momentum':
            self.vs = self.mu * self.vs - self.gradStep * dJs

        elif self.learning == 'NAG':
            self.vs = -(self.mu * self.vs) + (1. + self.mu) * (self.mu * self.vs - self.gradStep * dJs)

        elif self.learning == 'standard':
            self.vs = - self.gradStep * dJs

        elif self.learning == 'adagrad':
            self.cache = self.cache + dJs**2
            self.vs = - self.gradStep * dJs / np.array([np.sqrt(cache + self.epsilon) for i,cache in enumerate(self.cache)])

        elif self.learning == 'RMSprop':
            self.cache = self.decay * self.cache + (1. - self.decay) * dJs**2
            self.vs = - self.gradStep * dJs / np.array([np.sqrt(cache + self.epsilon) for i,cache in enumerate(self.cache)])

        elif self.learning == 'adam':
            self.cachedJ = self.beta1 * self.cachedJ + (1. - self.beta1) * dJs
            self.cachev = self.beta2 * self.cachev + (1. - self.beta2) * dJs**2

            cachedJt = self.cachedJ / (1. - self.beta1**(t+1.))
            cachevt = self.cachev / (1. - self.beta2**(t+1.))

            self.vs = - self.gradStep * cachedJt / np.array([np.sqrt(cache + self.epsilon) for i,cache in enumerate(cachevt)])


    def train(self, X, y, iterate=1000):
        print 'Initial cost: ', self.costFunction(X,y)
        for i in xrange(iterate):
            self.backward(X,y,i)
            if int(i+1) % (iterate/10) == 0: print i+1, self.costFunction(X,y)
            #self.error_history = np.append(self.error_history, [[i, self.costFunction(X,y)]], axis=0)
            

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
                return (1. - np.tanh(z)**2)
        elif self.func=='ReLU':
            if (deriv==False):
                z = np.where(z > 0, z, z * 0.1)
                return z
            else:
                z = np.where(z > 0, 1, 0.1)
                return z

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
    X, y = sklearn.datasets.make_moons(200, noise=0.1, random_state=0)
    y = y.reshape(y.size,1).astype(float)

elif test == 3:
    X, y = sklearn.datasets.make_circles(200, noise=0.1, factor = 0.1, random_state=0)
    y = y.reshape(y.size,1).astype(float)

elif test == 4:
    X = np.array(([0,0], [0,1], [1,0], [1,1]), dtype=float)
    y = np.array(([0], [1], [1], [0]), dtype=float)

elif test == 5:
    X = np.linspace(-5*np.pi, 5*np.pi, 201)
    X = X.reshape(X.size,1).astype(float)
    y = np.sin(X)
    inputLayerSize = 1

#X, meanX, stdX = standardise(X)
#y, meany, stdy = standardise(y)
X,y = normalise(X,y)
NN = Neural_Network(inputLayerSize=inputLayerSize, hiddenLayerSize=500,
                    func='ReLU', learning='adam')
NN.train(X,y,iterate=100000)
#np.savetxt('error.ev', NN.error_history, header='#')

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
    #X = destandardise(X, meanX, stdX)

    plt.plot(X,y)
    plt.plot(X,y_predicted)
    plt.show()




class Neurons:
    nodes = []
    def __init__(self,n):
        self.nodes = n
    def getLength(self):
        return len(self.nodes)
    def get(i):
        return self.nodes[i]
    def toRow(self):
        return matrix(self.nodes).transpose()
    def toColumn(self):
        return matrix(self.nodes)

class Weights:
    w = []
    def __init__(self,n):
        self.w = n
    def getRows(self):
        return len(self.w)
    def getColumns(self):
        return len(self.w[0])
    def toMatrix(self):
        return matrix(self.w)

def sigmoid(x):
    return 1/(1+e^(-x))
def relu(x):
    return max(0,x)
def sigmoidPrime(x):
    return x*(1-x)
def reluPrime(x):
    if x > 0:
        return 1
    else:
        return 0
def gradientMultiply(a, b):
    a_rows = len(a)
    a_cols = len(a[0])
    b_rows = len(b)
    b_cols = len(b[0])
    if a_rows > 1 or b_rows > 1 or a_cols != b_cols:
        return None
    result = []
    for i in xrange(a_cols):
        result[0][i] = a[0][i]*b[0][i]
    return result

nodes = []
weightList = []

def clearWeights():
    weightList = []

def getWeights():
    w = [weightList[i] for i in xrange(len(weightList))]
    return w

def setWeights(w):
    clearWeights()
    for i in xrange(len(w)):
        weightList.append(w[i])

def feedForward(inputI, output, topology, isSigmoid, w):
    #input and output both have one row
    error = [0 for x in xrange(len(output[0]))]
    nodes.append(Neurons(inputI[0]))
    hidden = []
    if w is None: #random weights
        w = Weights(len(topology)-1)
        for i in xrange(len(w)):
            weights = [[random() for j in xrange(len(w))] for k in xrange(len(w))]
            w[i] = Weights(weights)
    for i in xrange(len(w)):
        weightList.append(w[i])
        weights = w[i].toMatrix()
        if i == 0:
            hidden = matrix(inputI)*weights
        else:
            hidden = matrix(hidden)*weights
        if isSigmoid[i]:
            for j in xrange(len(hidden[0])):
                hidden[0,j] = sigmoid(hidden[0][j])
        else:
            for j in xrange(len(hidden[0])):
                hidden[0,j] = relu(hidden[0][j])
        nodes.append(Neurons(hidden[0]))
        # if the network is trained, then just return the last layer
    for i in xrange(len(error)):
        error[i] = (output[0][i]-hidden[0][i])**2
    return error

def backProp(error, isSigmoidA):
    newWeights = []
    index = len(weightList)-1
    last = nodes[-1]
    nodes = nodes[:-1]
    derivatives = []
    if(isSigmoid(index)):
        for i in xrange(len(derivatives[0])):
            derivatives[0,i] = sigmoidPrime(last.get(i))
    else:
        for i in xrange(len(derivatives)):
            derivatives[0,i] = reluPrime(last.get(i))
    #print stuff
    gradients = []
    for i in xrange(len(error)):
        gradients[i][0] = error[i]*derivatives[0][i]
    #getting new weights between output and last hidden layer
    nextLayer = nodes[-1]
    gradients2 = matrix(gradients)
    delta = gradients2*nextLayer.toRow()
    prev = matrix(weightList[-1]) #not sure lol
    weightList = weightList[:-1]
    bago = prev - delta.transpose()
    newWeights[index] = Weights(bago)
    index = index - 1
    #getting new weights between the other layers
    gradients2 = gradients2.transpose()
    while not (weightList == []):
        last = nextLayer
        nodes = nodes[-1]
        derivatives = []
        if isSigmoid[index]:
            for i in xrange(len(derivatives[0])):
                derivatives[0,i] = sigmoidPrime(last.get(i))
        else:
            for i in xrange(len(derivatives)):
                derivatives[0,i] = reluPrime(last.get(i))
        gradients2 = gradients2*(prev.transpose())
        gradients2 = gradientMultiply(gradients2,derivatives)
        nextLayer = nodes[-1]
        z = nextLayer.toColumn()
        prev = weightList[-1].toMatrix()
        weightList = weightList[:-1]
        delta = z*gradients2
        bago = prev - delta
        newWeights[index] = Weights(bago)
    return newWeights

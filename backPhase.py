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

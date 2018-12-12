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

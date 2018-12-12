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

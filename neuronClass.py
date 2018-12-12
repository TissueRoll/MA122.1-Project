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

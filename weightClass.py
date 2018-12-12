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

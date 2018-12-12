in1 = [[0.2,0.1,0.3,0.5]]
out1 = [[1,0,0]]
act = [True,True]
top = [4,2,3]
wait1 = [[0.15,0.14],[0.02,0.24],[0.62,0.2],[0.34,0.25]]
wait2 = [[0.22,0.07,0.58],[0.59,0.55,0.77]]
bigat = [Weights(wait1), Weights(wait2)]
wth = feedForward(in1,out1,top,act,bigat)
print(wth)

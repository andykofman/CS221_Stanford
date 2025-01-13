import numpy as np

"""
Modeling: What we want to compute
"""

# points = [(2,4), (4,2) ]
# d = 1

# Generate Artificial data

true_w = np.array ([1,2,3,4,5])
d = len(true_w)
points = [ ]

for i in range (1000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    # print (x, y)
    points. append((x,y))

def F(w):
    return sum ((w*x - y)**2 for x, y in points) / len(points)


def dF(w):
    return sum (2*(w*x - y) * x for x, y in points)  / len(points)

#Gradient Descent

###########################################

"""
Algorithms: how we compute it
"""

def gradientDescent (F,dF, d):
        
    w = np.zeros(d)
    eta =0.01
    for t in range (500):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print ('iterariom {}: w = {}, F(w) = {}'.format(t, w, value))
gradientDescent(F, dF, d)
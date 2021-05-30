import numpy as np

class Axis:
    def __init__(self, axis_name, axis_dim):
        self.axis_name = axis_name
        self.axis_dim = axis_dim


def printResultBy(axis, res):
    d = "res["
    for i in range(axis.axis_dim):
        d += ":, "
    d += "idx]"
    print(d)
    # arr = np.array()
    for idx in range(res.shape[axis.axis_dim]):
        arr =eval(d)
        print(arr, arr.shape)



# scores = np.load('results.npy')
a = np.arange(48).reshape(2, 2, 3, 4)
print(a)
printResultBy(Axis("fold", 1), a)

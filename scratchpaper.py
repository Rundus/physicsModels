import numpy as np

a = np.array([[1,2,3],[5,6,7]])
b = np.array([1,2])
c = np.array([1,10])
print((a.T*b).T)
print(np.sum((a.T*b).T,axis=0))

# print(((a.T*b)/c).T)

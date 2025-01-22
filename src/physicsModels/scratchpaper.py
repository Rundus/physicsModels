import numpy as np
a = np.array([10,10,10])
a[np.where(a >20)[0]] = 0
print(a)


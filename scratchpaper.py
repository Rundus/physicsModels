import numpy as np
from collections import Counter
a = [1,  3, 2, 3, 4, 6, 6, 8,9, 9,  9, 9,   9]
b = [11,12,13,14,15,16,17,  11,12, 13, 14, 15,16]


def removeDuplicates(a,b):
    from collections import defaultdict
    D = defaultdict(list)
    for i,item in enumerate(a):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}
    badIndicies = [D[key][1:] for key in D.keys()]
    badIndicies = [item for sublist in badIndicies for item in sublist]
    newA = np.delete(a,badIndicies,axis=0)
    newB = np.delete(b,badIndicies,axis=0)
    return a,b
print(newA)
print(newB)


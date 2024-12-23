import numpy as np

from itertools import product

a = [1,2,3]
b = ['a','b','c']
c = [10,11,12]
d = ['i','j','k']
combos1 = [list(thing) for thing in product(a,c)]
combos2 = [list(thing) for thing in product(b,d)]


print(combos1)
print(combos2)


# print(((a.T*b)/c).T)

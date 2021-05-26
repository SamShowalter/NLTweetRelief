import numpy as np
import itertools

a = [[1,2,3],[1,2,3]]

print(list(itertools.chain.from_iterable(a)))

print(a + a + [])

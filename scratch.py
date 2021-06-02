import numpy as np
import pandas as pd
import itertools

a = [[1,2,3],[1,2,3]]

print(list(itertools.chain.from_iterable(a)))

a = {"a":{"b":1, "c":2},
        "d":{"b":7, "c":4}}
print(pd.DataFrame(a))
print(a + a + [])

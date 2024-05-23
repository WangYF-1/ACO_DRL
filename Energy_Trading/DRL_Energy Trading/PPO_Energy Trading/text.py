import numpy as np
import pandas as pd
from random import *
import json
e_all = []

for p in range(300000):
    e_init = []
    e_exp = []
    for i in range(5):
        e_init.append(randint(10, 20))
        e_exp.append(randint(90, 100))
    for i in range(5):
        e_init.append(randint(90, 100))
        e_exp.append(randint(10, 20))
    e = [e_init, e_exp]
    e_all.append(e)
with open('ev.json', 'w') as file:
    json.dump(e_all, file)


















import numpy as np
import ast

import matplotlib.pyplot as plt

filename = "1.txt"
history = {}
with open(filename, "r") as f:
    history = ast.literal_eval(f.read())
fig = plt.figure(figsize=(6, 6))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
fig.show()
plt.show()


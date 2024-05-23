import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class painter:
    def __init__(self):
        self.data = None
    def smooth(self, rewards, weight):
        scalar = rewards
        last = scalar[0]
        smoothed = []
        weight = weight
        x = 'episode'
        y = 'reward'
        for point in scalar:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        ax = range(30, 50000, 30)
        save = pd.DataFrame({x: ax, y: smoothed})
        return save
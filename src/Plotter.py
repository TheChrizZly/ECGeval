import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class Plotter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y)
        plt.show()
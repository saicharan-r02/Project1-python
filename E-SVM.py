import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
l=load_digits()
plt.gray() 
for i in range(5):
    plt.matshow(l.images[i],cmap='inferno')    
plt.show()   
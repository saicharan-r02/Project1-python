from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
d = load_digits()
plt.gray() 
for i in range(5):
    plt.matshow(d.images[i],cmap='jet') 
plt.show()    
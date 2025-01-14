import numpy as np 
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x_values = np.linspace(-50,50,500)

s_values = sigmoid(x_values)

# plot the results
plt.figure(figsize=(10,5))
plt.plot(x_values, s_values, label='sigmoid')
plt.title('Sigmoid Function Plot')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the range of 'a'
a = np.linspace(-5, 5, 400)
y = 0.9**(a + 1)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(a, y, label='y = 0.9^(a+1)')
plt.title('Graph of y = 0.9^(a+1)')
plt.xlabel('a')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()

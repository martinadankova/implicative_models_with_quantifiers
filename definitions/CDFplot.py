import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Uniform distribution parameters
a = 0
b = 1

# Normal distribution parameters
mu = 0
sigma = 1

# Create x values
x_uniform = np.linspace(-0.5, 1.5, 1000)
x_normal = np.linspace(-4, 4, 1000)

# Calculate CDFs
cdf_uniform = np.piecewise(x_uniform, [x_uniform < a, (a <= x_uniform) & (x_uniform <= b), x_uniform > b],
                           [0, lambda x: (x - a) / (b - a), 1])
cdf_normal = norm.cdf(x_normal, loc=mu, scale=sigma)

# Plot CDFs
plt.figure(figsize=(12, 6))

# Uniform CDF plot
plt.subplot(1, 2, 1)
plt.plot(x_uniform, cdf_uniform, label="Uniform(0, 1)")
plt.xlabel('x')
plt.ylabel('F_X(x)')
plt.title('CDF of Uniform(0, 1)')
plt.legend()
plt.grid(True)

# Normal CDF plot
plt.subplot(1, 2, 2)
plt.plot(x_normal, cdf_normal, label="N(0, 1)")
plt.xlabel('x')
plt.ylabel('F_X(x)')
plt.title('CDF of N(0, 1)')
plt.legend()
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()

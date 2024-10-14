import numpy as np
import matplotlib.pyplot as plt

# Define the complex function f(z) = z^2
def f(z):
    return z**2

# Create a grid of points in the complex plane
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # Create complex grid

# Evaluate the function on the grid
F = f(Z)

# Calculate magnitude and phase
magnitude = np.abs(F)
phase = np.angle(F)

# Plot the magnitude
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, magnitude, levels=50, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('Magnitude of $f(z) = z^2$')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.grid(True)

# Plot the phase
plt.subplot(1, 2, 2)
plt.contourf(X, Y, phase, levels=50, cmap='twilight')
plt.colorbar(label='Phase (radians)')
plt.title('Phase of $f(z) = z^2$')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.grid(True)

plt.tight_layout()
plt.show()
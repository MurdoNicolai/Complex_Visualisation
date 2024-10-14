import numpy as np
import matplotlib.pyplot as plt

# Define the inverse function z = sqrt(f), so that f(z) = z^2
def inverse_f(F):
    return np.sqrt(F)

# Create a grid of points for real and imaginary parts of f(z)
real_f = np.linspace(-4, 4, 400)
imag_f = np.linspace(-4, 4, 400)
X_f, Y_f = np.meshgrid(real_f, imag_f)
F = X_f + 1j * Y_f  # Complex numbers representing f(z)

# Evaluate the inverse function (which gives z)
Z = inverse_f(F)

# Calculate the magnitude and phase of the corresponding z values
magnitude_z = np.abs(Z)
phase_z = np.angle(Z)

# Plot the magnitude of z
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X_f, Y_f, magnitude_z, levels=50, cmap='viridis')
plt.colorbar(label='Magnitude of $z$')
plt.title('Magnitude of $z = f^{-1}(f(z)) = \sqrt{f}$')
plt.xlabel('Real part of $f(z)$')
plt.ylabel('Imaginary part of $f(z)$')
plt.grid(True)

# Plot the phase of z
plt.subplot(1, 2, 2)
plt.contourf(X_f, Y_f, phase_z, levels=50, cmap='twilight')
plt.colorbar(label='Phase of $z$ (radians)')
plt.title('Phase of $z = f^{-1}(f(z)) = \sqrt{f}$')
plt.xlabel('Real part of $f(z)$')
plt.ylabel('Imaginary part of $f(z)$')
plt.grid(True)

plt.tight_layout()
plt.show()

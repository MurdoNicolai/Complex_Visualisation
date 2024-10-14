import numpy as np
import matplotlib.pyplot as plt
import math

import numpy as np
import matplotlib.pyplot as plt

# Define the grid transform function with f as input
def grid_transforme(f, xspace, yspace):
    # Define the grid of points in the complex plane
    x = np.linspace(xspace[0], xspace[1], 301)
    y = np.linspace(yspace[0], yspace[1], 1)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Combine into complex numbers

    # Apply the function f to each point in the grid
    FZ = np.empty_like(Z, dtype=complex)  # Create an empty array to hold f(z)
    None_points = []  # List to track None points

    # Fill FZ with transformed values and handle None
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            result = f(Z[i, j])
            if result is None:
                None_points.append((Z[i, j].real, Z[i, j].imag))  # Track where f(z) is None
                FZ[i, j] = np.nan  # Mark as NaN to exclude from plotting
            else:
                FZ[i, j] = result

    # Define a few special points to track (you can change these)
    special_points_z = np.array([1.001 + 10j, 1.002 + 10j, 1.003 + 10j, 1.005 + 10j, 1.01 + 10j])

    # Apply the function f to the special points, handling None
    special_points_fz = []
    for point in special_points_z:
        result = f(point)
        if result is None:
            special_points_fz.append(np.nan + 1j * np.nan)  # Mark as NaN for None
        else:
            special_points_fz.append(result)
    special_points_fz = np.array(special_points_fz)

    # Colors for the special points
    colors = ['r', 'g', 'b', 'm', 'c']

    # Plot the original grid lines (in the z-plane)
    plt.figure(figsize=(10, 5))

    # Plot in the z-plane
    plt.subplot(1, 2, 1)
    plt.plot(X, Y, 'b', alpha=0.5)  # Horizontal lines
    plt.plot(X.T, Y.T, 'b', alpha=0.5)  # Vertical lines
    plt.scatter(special_points_z.real, special_points_z.imag, color=colors, s=100, label="Special Points")  # Colored points
    plt.scatter([p[0] for p in None_points], [p[1] for p in None_points], color='black', s=100, marker='x', label="None Points")  # Mark None points
    plt.title("Original grid (z-plane)")
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.grid(True)
    plt.legend()

    # Plot in the f(z)-plane
    plt.subplot(1, 2, 2)
    plt.plot(FZ.real, FZ.imag, 'r', alpha=0.5)  # Transformed horizontal lines
    plt.plot(FZ.real.T, FZ.imag.T, 'r', alpha=0.5)  # Transformed vertical lines
    plt.scatter(special_points_fz.real, special_points_fz.imag, color=colors, s=100, label="Transformed Points")  # Colored points after transformation
    plt.scatter(FZ.real[np.isnan(FZ)], FZ.imag[np.isnan(FZ)], color='black', s=100, marker='x', label="None Transformed Points")  # Mark None points
    plt.title("Transformed grid (f(z)-plane)")
    plt.xlabel('Real part of f(z)')
    plt.ylabel('Imaginary part of f(z)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




def f(z):
    if z == 0:
        return None
    return np.exp(z)


def rieman_zeta(z):
    n = 1
    result = 0
    y = 1 + 1j
    while np.abs(y) > 0.001: 
        if n > 1000 or result > 40: 
            return None
        y = 1/pow(n, z)
        n += 1
        result += y 
    return (result)


def zeta_steps_plot(stepspace, z):
    realpart = []
    imaginary = []
    n = 1
    result = 0
    prevy = 1 + 0j
    prevang = 0
    while n < stepspace[1]: 
        y = 1/pow(n, z)
        ang = np.angle(y) - np.angle(prevy)
        if n>1:
            print(ang/prevang)
            print(np.log((n+1)/n)/np.log((n)/(n-1)))
        prevang = ang
        prevy = y
        result += y
        if n == 1 :       
            start = (result.real, result.imag)
        if n >= stepspace[0]:
            realpart.append(result.real)
            imaginary.append(result.imag)
        n += 1
    end = (result.real, result.imag)
    plt.plot(realpart, imaginary, 'r', alpha=0.5)
    plt.scatter([start[0], end[0]] ,[start[1], end[1]], color=['g', 'r'], s=100, label="Transformed Points")
    plt.title("Transformed grid (f(z)-plane)")
    plt.axis('equal')
    plt.xlabel('Real part of f(z)')
    plt.ylabel('Imaginary part of f(z)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

x = (1.001, 1.01)
y = (10, 10)
# grid_transforme(rieman_zeta, x, y )


stepspace = (0, 10)
zetastep_number = 1 + np.pi*1j
zeta_steps_plot(stepspace, zetastep_number)







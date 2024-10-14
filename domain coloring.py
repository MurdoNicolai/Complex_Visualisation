import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def domain_coloring(f, x_range=(-2, 2), y_range=(-2, 2), resolution=400):
    """
    Plots the domain coloring of a complex function f(z) with a phase mini-map.
    
    Parameters:
    - f: a complex function to visualize (e.g., lambda z: z**2).
    - x_range: tuple (xmin, xmax) for real axis.
    - y_range: tuple (ymin, ymax) for imaginary axis.
    - resolution: number of points along each axis.
    """
    # Create a grid of points in the complex plane
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Create complex numbers grid

    # Evaluate the function f(z) on the grid
    F = f(Z)

    # Calculate magnitude and phase of f(z)
    magnitude = np.abs(F)
    phase = np.angle(F)

    # Normalize the magnitude for brightness
    magnitude = np.log(1 + magnitude)

    # Create the color map: Phase controls hue, magnitude controls brightness
    H = (phase + np.pi) / (2 * np.pi)  
    S = np.ones_like(magnitude)        
    V = magnitude / np.max(magnitude)  

    # Stack H, S, V channels together and convert HSV to RGB
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)

    # Create the main domain coloring plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})

    # Main domain coloring plot
    ax[0].imshow(RGB, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower')
    ax[0].set_title(f'Domain Coloring of {f.__name__}')
    ax[0].set_xlabel('Real part of $z$')
    ax[0].set_ylabel('Imaginary part of $z$')
    ax[0].grid(False)

    # Create the phase mini-map (-1 - i to 1 + i) with constant brightness
    phase_x = np.linspace(-1, 1, 200)
    phase_y = np.linspace(-1, 1, 200)
    X_phase, Y_phase = np.meshgrid(phase_x, phase_y)
    Z_phase = X_phase + 1j * Y_phase

    H_phase = (np.angle(Z_phase) + np.pi) / (2 * np.pi)  
    S_phase = np.ones_like(H_phase)                      
    V_phase = np.ones_like(H_phase)                      

    HSV_phase = np.dstack((H_phase, S_phase, V_phase))
    RGB_phase = hsv_to_rgb(HSV_phase)

    # Plot the phase mini-map
    ax[1].imshow(RGB_phase, extent=[-1, 1, -1, 1], origin='lower')
    ax[1].set_title("Phase Map ($-1-i$ to $1+i$)")
    ax[1].set_xlabel('Real part of $z$')
    ax[1].set_ylabel('Imaginary part of $z$')
    ax[1].grid(False)

    plt.tight_layout()
    plt.show()

# Example usage with f(z) = z^2
domain_coloring(lambda z: z**2)

import pylab as plt
import numpy as np
def plot_caustic_analysis(result, Mu, crit_lines, caustics, xmin, xmax, ymin, ymax):
    """
    Plot the magnification field with caustic analysis visualization.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing caustic expansion results
    Mu : array
        Magnification field
    crit_lines : list
        Critical curves
    caustics : list
        Caustics
    xmin, xmax, ymin, ymax : float
        Plot boundaries
    """
    # Extract variables from result
    x0, y0 = result['x0'], result['y0']
    nabla_detA, nabla_detA_rot = result['nabla_detA'], result['nabla_detA_rot']
    v_per, v_par = result['v_per'], result['v_par']
    img = result['img']
    x_src, y_src = result['x_src'], result['y_src']
    
    # Plot the absolute value of the magnification, the image positions, and eigenvectors:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(Mu, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='inferno', vmin=-20, vmax=20) # Magnifications
    cbar = plt.colorbar(im)
    cbar.set_label('Magnification')
    # Plot eigenvectors from 0:
    plt.quiver(x0, y0, v_per[0], v_per[1], color='brown', scale=5, label=r"$\vec v_\text{per}$")
    plt.quiver(x0, y0, v_par[0], v_par[1], color='orange', scale=5, label=r"$\vec v_\text{par}$")
    # Plot the nabla_detA and the rotated nabla_detA vector from 0:
    plt.quiver(x0, y0, nabla_detA[0]/np.linalg.norm(nabla_detA), nabla_detA[1]/np.linalg.norm(nabla_detA), 
               color='white', scale=5, label=r"$\nabla D_0$")
    plt.quiver(x0, y0, nabla_detA_rot[0]/np.linalg.norm(nabla_detA_rot), nabla_detA_rot[1]/np.linalg.norm(nabla_detA_rot), 
               color='gray', scale=5, label=r"$R(\pi/2) \nabla D_0$")
    # Plot critical curves and caustics:
    plt.plot(crit_lines[0][0], crit_lines[0][1], color='cyan', lw=1, label='Critical Curves') # Critical curves
    plt.plot(caustics[0][0], caustics[0][1], color='lime', lw=1, label='Caustics') # Caustics
    # Plot expansion point:
    plt.scatter(x0, y0, color='blue', s=50, label='Point on Critical Curve', marker='o') # Point on critical curve
    # Plot the images and the source position
    plt.scatter(img[:,0], img[:,1], color='cyan', s=50, marker='x', label='Images (numerical)', alpha=1) # Images as crosses
    plt.scatter(x_src, y_src, color='purple', s=50, label='Source Position', marker='x') # Source position
    plt.xlabel(r"x ($\theta_E$)")
    plt.ylabel(r"y ($\theta_E$)")
    plt.title('Fold Caustic Lens Magnification')
    plt.legend(loc='upper left')
    plt.show()

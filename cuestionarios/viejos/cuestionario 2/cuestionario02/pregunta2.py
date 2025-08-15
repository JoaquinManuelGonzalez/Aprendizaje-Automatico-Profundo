import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi

np.seterr(all='raise')

# Define the function to plot
def f(x, y):
    z = x/(2*x**2+3*y**2+1)
    return z


# Define the derivatives
def df_dx(x, y):
    df_dx= (-2*x**2+3*y**2+1)/(2*x**2+3*y**2+1)**2
    return df_dx


def df_dy(x, y):
    df_dy= (-6*x*y)/(2*x**2+3*y**2+1)**2
    return df_dy


# gradient descent
def gradient_descent(x, y, iterations, alpha,ax):
    #global ax2,ax
    epsilon = 10e-6
    x_old = x + 1
    y_old = y + 1
    while iterations > 0 and abs(f(x, y) - f(x_old, y_old)) > epsilon:
        x_old = x
        y_old = y
        try:
            x = x - alpha * df_dx(x, y)
            y = y - alpha * df_dy(x, y)
            ax.scatter(x, y, f(x, y), c="b")
        except:
            #ax2=fig2.add_subplot(111)
            #ax=fig.add_subplot(111, projection="3d")
            return x_old, y_old
        iterations -= 1
    print("x = {}, y = {}, f(x,y) = {}".format(x, y, f(x, y)))
    if (abs(f(x, y) - f(x_old, y_old)) > epsilon):
        print("No se pudo encontrar un mínimo")
    return x, y


# Create the data to plot
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the figure and the 3D axes
iterations = 100
alpha = 0.1
fig = plt.figure()
fig2 = plt.figure()
def create_figures():
    # make 3d plot
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color="g", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # make other 2d plot to show the gradient descent with contour
    ax2 = fig2.add_subplot(111)
    ax2.contour(X, Y, Z, 50, cmap="viridis", antialiased=True, alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Gradient descent")
    cid = fig2.canvas.mpl_connect('button_press_event', onclick)
    return ax,ax2

# reset plots
def reset_plots():
    fig.clear()
    fig2.clear()
    ax,ax2 = create_figures()
    return ax,ax2

# allow touching fig2 to select the initial point
def onclick(event):
    global fig,ax,fig2,ax2
    x = event.xdata
    y = event.ydata
    print("x = {}, y = {}".format(x, y))
    # Plot the gradient descent
    try:
        x, y = gradient_descent(x, y, iterations, alpha,ax2)
        x, y = gradient_descent(x, y, iterations, alpha,ax)
    except:
        ax,ax2=reset_plots()

    print("x = {}, y = {}".format(x, y))
    plt.show()


# detect if the r key is pressed
def press(event):
    global fig,ax,fig2,ax2
    if event.key == 'r':
        ax,ax2=reset_plots()
        fig.canvas.draw()
        fig2.canvas.draw()

# connect the press event to the function press
fig.canvas.mpl_connect('key_press_event', press)
fig2.canvas.mpl_connect('key_press_event', press)

ax,ax2 = create_figures()

plt.show()

# Dylan Olthoff
# CST 305
# Project 2 Topic 2 – Runge-Kutta-Fehlberg (RKF) for ODE
# Dr. Ricardo Citro

# In this project we are comparing Runge-Kutta to ODEint
# implementation
# the functions for Runge-Kutta and ODEint using the imported packages
# got the solutions using a differential equation and then compared and graphed the results

# imported packages
import time                         # import time to use for performance analysis
import numpy as np                  # import numpy for array space
import matplotlib.pyplot as plt     # import matplotlib for graphing functions
from scipy.integrate import odeint  # import scipy to use the ordinary differencial equation integral function


def dy_dx(y, x):                    # get the input for y and x
    return (-(y) + np.log(x))     # returns the function answer


def RungeKutta(y, x, h):
    k1 = dy_dx(y, x)                                    # get k1 using differential equation
    k2 = dy_dx(y + ((0.5 * h) * k1), x + 0.5 * h)       # get k2 using k1 and differential equation
    k3 = dy_dx(y + ((0.5 * h) * k2), x + 0.5 * h)       # get k3 using k2 and differential equation
    k4 = dy_dx(y + (h * k3), x + h)                     # get k4 using k3 and differential equation


    y = y + ((1.0 / 6.0) * (k1 + (2 * k2) + (2 * k3) + k4) * h)   # get y by adding T4*h to the initial y
    return y

# initial values
y = 1
x = 2
h = 0.3
n = 2000

# arrays used to store the x and y values
xRunge = []      # x-space runge-kutta array to store the values of x for the runge-kutta function
yRunge = []      # y-space runge-kutta array to store the values of y for the runge-kutta function

executionODE = time.time() # tracks time for ODEint

xODE = np.linspace(1, (int)(n * h)+1, n) # x value space
yODE = odeint(dy_dx, y, xODE)  # y value space

endTime = time.time()  # end runtime for ODEint
ODEtime = endTime - executionODE # total time taken to run

# ODEint Graph
plt.title("ODE Function")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xODE, yODE, 'r-', label = "ODEint", linewidth = 2)
plt.legend()
plt.show()

# Runge-Kutta function
RungeTime = time.time() # start time for computation

for i in range(0, n): # runs the function n number of times
    xRunge.append(x) # add each x to the array
    yRunge.append(y) # add each y to the array
    y = RungeKutta(y, x, h) # change y value so it is ready for next iteration
    x = x + h # increment x value

endRunge = time.time()                               # time end for runge-kutta function solution
timeTaken = endRunge - RungeTime                     # total time the runge-kutta function to solve

print(xRunge)
print(yRunge)


# Runge-Kutta Graph
plt.title("Runge-Kutta Function Analysis")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xRunge, yRunge, 'b-', label = "Runge Kutta")
plt.legend()
plt.show()


# Times for each function
print("\nODEint Time:         ", ODEtime)
print("Runge Kutta Time:    ", timeTaken)

# final solutions
print("\nODEint answer: ", yODE[-1])
print("Runge-Kutta answer: ", yRunge[-1])

# Graph for both functions
plt.title("Runge-Kutta and ODE Function Analysis")          # set the title of the graph
plt.xlabel("x")                                             # set the x label on the graph
plt.ylabel("y")                                             # set the y label on the graph
plt.plot(xODE, yODE, 'r-', label = "ODEint", linewidth = 2)   # set the ODE line to be red and label it
plt.plot(xRunge, yRunge, 'b-', label = "Runge Kutta")             # set the runge-kutta to be blue and label it
plt.legend()                                                # shows the legend on the graph
plt.show()
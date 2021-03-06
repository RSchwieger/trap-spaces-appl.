__author__ = 'Robert Schwieger'

from scipy.integrate import ode
import matplotlib.pyplot as plt

# Constants:

k = [7,8,6,9] # Hill coefficients
theta = [0.5, 0.5, 0.7, 0.7] # thresholds
d = [2.5,1,1.5,0.5] # diagonal of D-Matrix
x0 = [0.55,0.25,0.9,0.0] # initial value (1,*,*,*)

# Functions of the ODE-system

def multivariateInterpolation(x):
    """
    Multivariate interpolation
    :param x: list of real numbers in the interval [0,1]
    :return: list of real numbers in the interval [0,1]
    """
    y = [0,0,0,0]
    y[0] = x[0]+x[1]-x[0]*x[1]
    y[1] = x[0]*x[3]
    y[2] = (1-x[0])*x[3]
    y[3] = 1-x[2]
    return y

def hillCube(x, k, theta):
    """
    Computes the Hill Cube at x with Hill coefficients k and threshold theta
    :param x: list of input values of the Hill Cube of length N
    :param k: list of Hill coefficients of length N
    :param theta: list of thresholds of length N
    :return: result of the Hill Cube evaluated at x represented as a list of length N
    """
    return [(x[i]**k[i])/(x[i]**k[i]+theta[i]**k[i]) for i in range(len(x))]

def f(t,x):
    """
    Function of the ODE-system written in the form required by the ODE-solver
    :param t: Not used
    :param x: list of input values
    :return: function evaluation saves as list
    """
    discreteTimestep = multivariateInterpolation(hillCube(x, k, theta))
    return [d[i]*(discreteTimestep[i]-x[i]) for i in range(len(discreteTimestep))]

# Constants for the ODE-solver

stoppingTime = 10.0
number_of_steps = 10**4 # Anzahl der Iterationen
dt = stoppingTime/number_of_steps # Schrittgröße

# Solving the ODE-system

y = ode(f).set_integrator('dopri5')
y.set_initial_value(x0, 0.0) # set initial value at time = 0
evaluationTimes = [0.0] # initialized
solution = [x0] # save the first time step

while y.successful() and y.t < stoppingTime:
    evaluationTimes += [y.t+dt]
    y.integrate(y.t+dt)
    solution += [list(y.y)]

    if y.successful() is False:
        print("Something went wrong during r.integrate()")

# Plot solution

plt.ion()
plt.axis([0.0, stoppingTime, 0.0, 1.1])
for i in range(len(x0)):
    componentOfSolution = [solution[j][i] for j in range(len(solution))] # extract i-th component of solution vector
    plt.plot(evaluationTimes, componentOfSolution, label='x'+str(i+1))

plt.ylabel('x')
plt.xlabel('time')
plt.legend(loc=0)
plt.title("Trajectory of the solutions of the ODE-system with initial state "+str(x0)+", d = "+str(d)+" ,theta = "+str(theta)+", k="+str(k))
plt.show(block=True)


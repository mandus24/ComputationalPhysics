import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def bootstrap_errors(y, num_samples=1000):
    """Performs bootstrap resampling to estimate errors."""
    n = len(y)
    boot_samples = np.random.choice(y, size=(num_samples, n), replace=True)
    means = np.mean(boot_samples, axis=1)
    return np.std(means)  # Standard deviation as error estimate

def fit_power_law(x, y, initial_exponent, fit_type):
    """Fits y = A * x^exponent to the data."""
    if fit_type == 'decay':
        func = lambda N, A, p: A * N**(-p)
    elif fit_type == 'growth':
        func = lambda N, A, nu: A * N**nu
    
    popt, _ = curve_fit(func, x, y, p0=[1.0, initial_exponent])
    return popt  # Returns fitted parameters

# Load the CSV file
data = pd.read_csv('3dparam.csv')

x = data.iloc[:, 0].values
y1 = data.iloc[:, 1].values
y2 = data.iloc[:, 2].values
y3 = data.iloc[:, 3].values

# Compute bootstrap errors
error_y1 = bootstrap_errors(y1)
error_y2 = bootstrap_errors(y2)
error_y3 = bootstrap_errors(y3)

# Fit models
A1, p = fit_power_law(x, y1, 1.0, 'decay')
A2, nu1 = fit_power_law(x, y2, 0.59, 'growth')
A3, nu2 = fit_power_law(x, y3, 0.59, 'growth')  # Reuse nu fit

x_fit = np.linspace(min(x), max(x), 100)

# Plot y1
plt.figure(figsize=(16,10))
plt.errorbar(x, y1, yerr=error_y1, fmt='o', label='Data', color='b')
plt.plot(x_fit, A1 * x_fit**(-p), 'b-', label=f'Fit: f = {A1:.3f} * N^(-{p:.3f})')
plt.xlabel('N')
plt.ylabel('f')
plt.rcParams.update({'font.size':25})
plt.title('Acceptance rate')
plt.legend()
plt.show()

# Plot y2
plt.figure(figsize=(16,10))
plt.errorbar(x, y2, yerr=error_y2, fmt='s', label='Data', color='r')
plt.plot(x_fit, A2 * x_fit**nu1, 'r-', label=f'Fit: <ω²> = {A2:.3f} * N^{nu1:.3f}')
plt.xlabel('N')
plt.ylabel('<ω²>')
plt.rcParams.update({'font.size': 25})
plt.title('End-to-end distance')
plt.legend()
plt.show()

# Plot y3
plt.figure(figsize=(16,10))
plt.errorbar(x, y3, yerr=error_y3, fmt='^', label='Data', color='g')
plt.plot(x_fit, A3 * x_fit**nu2, 'g-', label=f'Fit: <S²> = {A3:.3f} * N^{nu2:.3f}')
plt.xlabel('N')
plt.ylabel('<S²>')
plt.rcParams.update({'font.size': 25})
plt.title('Gyration')
plt.legend()
plt.show()

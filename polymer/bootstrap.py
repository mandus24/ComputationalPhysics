import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
from scipy.optimize import curve_fit

def polynomial(x,a,b):
    return a*x**b

Ns = [101,201,401,601,801,1001,1201,1401,1601,2001,2401,3001,4001,5001,6001,7001,8001,9001,10001]

ω_squareds = np.concatenate((np.load("results_dimerisation/ω_squareds.npy"), np.load("results_thermalisation/ω_squareds.npy")[:,10**5:]))
S_squareds = np.concatenate((np.load("results_dimerisation/S_squareds.npy"), np.load("results_thermalisation/S_squareds.npy")[:,10**5:]))

error_ω_squareds = np.concatenate((np.load("results_dimerisation/error_ω_squareds.npy"), np.load("results_thermalisation/error_ω_squareds.npy")))
error_S_squareds = np.concatenate((np.load("results_dimerisation/error_S_squareds.npy"), np.load("results_thermalisation/error_S_squareds.npy")))

autocorr_time_ω_squareds = np.concatenate((np.load("results_dimerisation/autocorr_time_ω_squareds.npy") , np.load("results_thermalisation/autocorr_time_ω_squareds.npy")))
autocorr_time_S_squareds = np.concatenate((np.load("results_dimerisation/autocorr_time_S_squareds.npy") , np.load("results_thermalisation/autocorr_time_S_squareds.npy")))




def bootstrap_analysis_prep(data,auto_time):
    n = len(data)
    iid_length = n//(2*int(auto_time))
    iid_data = []
    for i in range(iid_length):
        if (i == iid_length-1):
            iid_data.append(np.mean(data[i*2*int(auto_time):]))
        else:
            iid_data.append(np.mean(data[i*2*int(auto_time):(i+1)*2*int(auto_time)]))
    return iid_data
def bootstrap_analysis(iid_data, B):
    iid_length = len(iid_data)
    exp_values = []
    for b in range(B):
        bootstrap_sample = []
        for i in range(iid_length):
            index = np.random.randint(0,iid_length)
            bootstrap_sample.append(iid_data[index])
        exp_values.append(np.mean(bootstrap_sample))

    return exp_values

def parameter_errors(data, autocorrelation_times, Ns, iterations = 1000):
    B = 100
    exp_values = np.zeros((len(Ns),iterations))
    err_values = np.zeros((len(Ns),iterations))
    all_mean_values = np.zeros((len(Ns),iterations*B))
    for i,N in enumerate(Ns):
        print(f"N = {N}")
        iid_data = bootstrap_analysis_prep(data[i,:], autocorrelation_times[i])
        for iter in range(iterations):
            tmp = bootstrap_analysis(iid_data, B)
            exp_values[i,iter] = np.mean(tmp)
            err_values[i,iter] = np.std(tmp)
            all_mean_values[i,iter*B:(iter+1)*B] = tmp
        
    As = []
    exponents = []
    for iter in range(iterations):
        #print(f"Iteration = {iter}")
        popt, pcov = curve_fit(polynomial, Ns, exp_values[:,iter], sigma = err_values[:,iter],absolute_sigma=True)
        As.append(popt[0])
        exponents.append(popt[1])

    return np.mean(As), np.std(As), np.mean(exponents), np.std(exponents),np.mean(all_mean_values,axis=1), np.std(all_mean_values,axis=1)

A_mean_ω, A_std_ω, exponent_mean_ω, exponent_std_ω, exp_values_ω, err_values_ω = parameter_errors(ω_squareds, autocorr_time_ω_squareds, Ns, 100)
A_mean_S, A_std_S, exponent_mean_S, exponent_std_S, exp_values_S, err_values_S = parameter_errors(S_squareds, autocorr_time_S_squareds, Ns, 100)

np.save("bootstrap/A_mean_ω.npy",A_mean_ω)
np.save("bootstrap/A_std_ω.npy",A_std_ω)
np.save("bootstrap/exponent_mean_ω.npy",exponent_mean_ω)
np.save("bootstrap/exponent_std_ω.npy",exponent_std_ω)
np.save("bootstrap/exp_values_ω.npy",exp_values_ω)
np.save("bootstrap/err_values_ω.npy",err_values_ω)
np.save("bootstrap/A_mean_S.npy",A_mean_S)
np.save("bootstrap/A_std_S.npy",A_std_S)
np.save("bootstrap/exponent_mean_S.npy",exponent_mean_S)
np.save("bootstrap/exponent_std_S.npy",exponent_std_S)
np.save("bootstrap/xp_values_S.npy",exp_values_S)
np.save("bootstrap/err_values_S.npy",err_values_S)
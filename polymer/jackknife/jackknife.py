import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm
import sys


def autocorrelation(timeseries, mean=None):
    timeseries = np.array(timeseries)
    if mean is None:
        mean = timeseries.mean()
    fluctuations = timeseries-mean

    C = np.fft.ifft(np.fft.fft(fluctuations) * np.fft.ifft(fluctuations)).real
    return C/C[0]

def integrated_autocorrelation_time(timeseries, mean=None, until=None):
    steps = len(timeseries)
    if until is None:
        until = steps // 2
    Gamma = autocorrelation(timeseries, mean=mean)[:until]
    try:
        first_zero = np.where(Gamma <= 0)[0][0]
        #print(first_zero)
        return 0.5 + Gamma[1:first_zero].sum()
    except:
        # Gamma never hits 0.  So the autocorrelation spans ~ the whole ensemble.
        return steps # at least!




def jackknife_sampling(data,τ):
    τ_jack = []
    M = len(data)
    τ = int(np.ceil(τ))
    block_size = 2 * τ
    num_blocks = M // block_size
    
    for i in range(num_blocks):
        tmp_data = np.delete(data, np.s_[i * block_size: (i + 1) * block_size])
        τ_jack.append(integrated_autocorrelation_time(tmp_data))
    τ_mean = np.mean(τ_jack)
    τ_var = (num_blocks - 1) / num_blocks * np.sum((τ_jack - τ_mean) ** 2)
    return np.mean(τ_jack), np.sqrt(τ_var)





Ns = [101,201,401,601,801,1001,1201,1401,1601,2001,2401,3001,4001,5001,6001,7001,8001,9001,10001]
ω_squareds = np.concatenate((np.load("../../results_dimerisation/ω_squareds.npy"), np.load("../../results_thermalisation/ω_squareds.npy")[:,10**5:]))
S_squareds = np.concatenate((np.load("../../results_dimerisation/S_squareds.npy"), np.load("../../results_thermalisation/S_squareds.npy")[:,10**5:]))

index = int(sys.argv[1])

ω_squared = ω_squareds[index,:]
S_squared = S_squareds[index,:]
autocorr_time_ω_squareds_first0 = integrated_autocorrelation_time(ω_squared)
autocorr_time_S_squareds_first0= integrated_autocorrelation_time(S_squared)
    

τ_mean_ω, τ_error_ω = jackknife_sampling(ω_squared, autocorr_time_ω_squareds_first0)
τ_mean_S, τ_error_S = jackknife_sampling(S_squared, autocorr_time_S_squareds_first0)

with open("omega.txt", 'w') as fω, open("S.txt", 'w') as fS:
    fω.write(f"{τ_mean_ω}\t{τ_error_ω}\n")
    fS.write(f"{τ_mean_S}\t{τ_error_S}\n")
 

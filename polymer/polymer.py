import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm


def check_self_avoidance(coords, pivot = None):
    # False = Self avoidant
    # True = not Self avoidant

    if pivot == None:
        N = np.size(coords, axis=1)
        hashset = set()
        for i in range(N):
            tmp_tuple = tuple(coords[:,i])
            if tmp_tuple in hashset:
                return True
            else:
                hashset.add(tmp_tuple)


        return False

    else:
        N = np.size(coords, axis=1)
        if pivot >= N//2:
            longer_site= pivot
            shorter_site= N-pivot-1
            dir = -1
        else:
            longer_site = N-pivot-1
            shorter_site = pivot
            dir = 1

        hashset=set()
        for i in range(1,longer_site+1):
            if i > shorter_site:   # Case when we scan only in the longer direction from the pivot because the short direction was already fully scanned
                tmp_tuple = tuple(coords[:,pivot+i*dir])
                if tmp_tuple in hashset:
                    return True
                else:
                    hashset.add(tmp_tuple)
            else: # Check in both directions of the pivot
                tmp_tuple = tuple(coords[:,pivot+i])
                if tmp_tuple in hashset:
                    return True
                else:
                    hashset.add(tmp_tuple)
                tmp_tuple = tuple(coords[:,pivot-i])
                if tmp_tuple in hashset:
                    return True
                else:
                    hashset.add(tmp_tuple)
        return False


def generate_random_walk(L):
    coord = np.zeros((2,L),dtype=int)
    for l in range(1,L):
        direction = np.random.randint(0,4) # clock wise starting from the top
        if direction == 0:
            coord[0,l] = coord[0,l-1]
            coord[1,l] = coord[1,l-1]+1

        elif  direction == 1:
            coord[0,l] = coord[0,l-1]+1
            coord[1,l] = coord[1,l-1]

        elif  direction == 2:
            coord[0,l] = coord[0,l-1]
            coord[1,l] = coord[1,l-1]-1

        elif  direction == 3:
            coord[0,l] = coord[0,l-1]-1
            coord[1,l] =  coord[1,l-1]

    return coord


def initial_SAW(N: int): #Dimerisation
    if N <= 5:
        tmp = True
        while tmp:
            walk = generate_random_walk(N)
            tmp = check_self_avoidance(walk)
        return walk



    else:
        tmp = True
        while tmp:
            first = initial_SAW(N//2)
            second = initial_SAW(N-N//2+1)
            translated = (first[:,-1]+second.T).T
            concatenated = np.concatenate((first,translated[:,1:]),axis=1)
            tmp = check_self_avoidance(concatenated)

        return concatenated



def symmetry_op(coord):
    N = np.size(coord,axis=1)
    g = np.random.randint(0,7)
    if g==0: #+90°(mathematically)
        for i in range(N):
            coord[:,i] = np.dot(np.array([[0,-1],[1,0]]), coord[:,i])
    elif g==1: #-90°(mathematically)
        for i in range(N):
            coord[:,i] = np.dot(np.array([[0,1],[-1,0]]), coord[:,i])
    elif g==2: #180°
        for i in range(N):
            coord[:,i] = np.dot(np.array([[-1,0],[0,-1]]), coord[:,i])
    elif g==3: #x-axis symmetry
        for i in range(N):
            coord[:,i] = np.dot(np.array([[1,0],[0,-1]]), coord[:,i])
    elif g==4: #y-axis symmetry
        for i in range(N):
            coord[:,i] = np.dot(np.array([[-1,0],[0,1]]), coord[:,i])
    elif g==5: #diagonal reflection upperright-bottomleft
        for i in range(N):
            coord[:,i] = np.dot(np.array([[0,1],[1,0]]), coord[:,i])
    elif g==6: #diagonal reflection upperleft-bottomright
        for i in range(N):
            coord[:,i] = np.dot(np.array([[0,-1],[-1,0]]), coord[:,i])
    if debug:
        print("g = ",g)
    return coord, g



def pivot_step(coord):
    global accepted
    N = np.size(coord,axis=1)
    pivot = np.random.randint(1,N)
    if debug:
        print("pivot = ",pivot)
    to_be_operated = coord[:,pivot:]
    to_be_operated = (to_be_operated.T-to_be_operated[:,0]).T
    operated, g = symmetry_op(to_be_operated)
    accepted[1,g] +=1
    operated = (operated.T+coord[:,pivot]).T

    coord_new = np.concatenate((coord[:,:pivot],operated),axis=1)
    not_selfavoidant = check_self_avoidance(coord_new,pivot)
    if not_selfavoidant:
        if debug:
            print("Not selfavoidant")
        return coord
    else:
        accepted[0,g] +=1
        return coord_new





def squared_end_to_end(coord):
    return np.linalg.norm(coord[:,-1])**2

def squared_gyration(coord):
    N = np.size(coord,axis=1)
    a = (1/N * np.sum(coord,axis=1))
    inbracket = np.dot(a,a)
    result = 0
    for i in range(N):
        result += np.dot(coord[:,i], coord[:,i])-inbracket
    return 1/N * result




def pivot_run(coord,iterations):
    global accepted
    ω_squared = []
    S_squared = []
    for i in tqdm(range(iterations)):
        coord = pivot_step(coord)
        ω_squared.append(squared_end_to_end(coord))
        S_squared.append(squared_gyration(coord))
    return ω_squared, S_squared, accepted




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
        print("First zero = ",first_zero)
        return 0.5 + Gamma[1:first_zero].sum()
    except:
        # Gamma never hits 0.  So the autocorrelation spans ~ the whole ensemble.
        return steps # at least!


def binning_analyis(data,kmax):
    M = len(data)
    error_est = []
    for k in range(1,kmax+1):
        M_k = M//k
        X_k = np.mean(np.reshape(data[:M_k*k],(-1,k)).T,axis=0)
        #print(len(X_k))
        error_est.append(np.std(X_k)/np.sqrt(M_k))
    τ = 0.5*((error_est[-1]/error_est[0])**2-1)
    return error_est, τ








#Ns = [101,201,401,601,801,1001,1201,1401,1601,2001,2401,3001,4001]
#seeds = np.arrayarray([2781300983, 552437507, 239163331, 1112836520, 1562221245, 4039863786,  930428319, 4185896816, 2716273418, 4265127893, 432287740, 2672218962, 3546297565])
Ns = [16,21]
seeds = np.array([2790077416, 1086981728])
iterations = 10**6
binning_max = 200
times = np.zeros((2,len(Ns)))
ω_squareds = np.zeros((len(Ns),iterations))
S_squareds = np.zeros((len(Ns),iterations))
error_ω_squareds = np.zeros((len(Ns),binning_max))
error_S_squareds = np.zeros((len(Ns),binning_max))
autocorr_time_ω_squareds = np.zeros(len(Ns))
autocorr_time_S_squareds = np.zeros(len(Ns))
debug = 0

for i,N in enumerate(Ns):
    np.random.seed(seeds[i])
    accepted = np.zeros((2,7))
    start = time.time()
    coord = initial_SAW(N)
    end = time.time()
    ω_squared, S_squared, accepted = pivot_run(coord,iterations)
    end_total = time.time()
    ω_squareds[i,:] = ω_squared
    S_squareds[i,:] = S_squared

    times[0,i] = end-start
    times[1,i] = end_total-start

    np.save(f"accepted{N}.npy",accepted)

    error_ω_squared, autocorr_time_ω_squared = binning_analyis(ω_squared,binning_max)
    error_S_squared, autocorr_time_S_squared = binning_analyis(S_squared,binning_max)
    error_ω_squareds[i,:] = error_ω_squared
    error_S_squareds[i,:] = error_S_squared
    autocorr_time_ω_squareds[i] = autocorr_time_ω_squared
    autocorr_time_S_squareds[i] = autocorr_time_S_squared
    

    plt.xlabel("Binning level")
    plt.ylabel("Error")
    plt.grid()
    plt.scatter(range(binning_max),error_ω_squared,label="Squared end-to-end distance")
    plt.scatter(range(binning_max),error_S_squared,label="Squared radius of gyration")
    plt.legend()
    plt.savefig(f"Binning_analysis{N}.png")
    plt.close()


np.save("ω_squareds.npy",ω_squareds)
np.save("S_squareds.npy",S_squareds)
np.save("times.npy",times)
np.save("error_ω_squareds.npy",error_ω_squareds)
np.save("error_S_squareds.npy",error_S_squareds)
np.save("autocorr_time_ω_squareds.npy",autocorr_time_ω_squareds)
np.save("autocorr_time_S_squareds.npy",autocorr_time_S_squareds)


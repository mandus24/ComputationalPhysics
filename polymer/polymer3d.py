import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm

def check_self_avoidance(coords, pivot=None):
    # False = Self avoidant
    # True = not Self avoidant

    if pivot is None:
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
            longer_site = pivot 
            shorter_site = N-pivot-1
            dir = -1
        else:
            longer_site = N-pivot-1
            shorter_site = pivot
            dir = 1
       
        hashset = set()
        for i in range(1, longer_site+1):
            if i > shorter_site:   # Case when we scan only in the longer direction from the pivot because the short direction was already fully scanned
                tmp_tuple = tuple(coords[:, pivot+i*dir])
                if tmp_tuple in hashset:
                    return True
                else:
                    hashset.add(tmp_tuple)
            else: # Check in both directions of the pivot
                tmp_tuple = tuple(coords[:, pivot+i])
                if tmp_tuple in hashset:
                    return True
                else: 
                    hashset.add(tmp_tuple)
                tmp_tuple = tuple(coords[:, pivot-i])
                if tmp_tuple in hashset:
                    return True
                else: 
                    hashset.add(tmp_tuple)
        return False
    
def generate_random_walk(L):
    coord = np.zeros((3, L), dtype=int)
    for l in range(1, L):
        direction = np.random.randint(0, 6)  # 6 directions in 3D
        if direction == 0:  # +x
            coord[0, l] = coord[0, l-1] + 1
            coord[1, l] = coord[1, l-1]
            coord[2, l] = coord[2, l-1]
        elif direction == 1:  # -x
            coord[0, l] = coord[0, l-1] - 1
            coord[1, l] = coord[1, l-1]
            coord[2, l] = coord[2, l-1]
        elif direction == 2:  # +y
            coord[0, l] = coord[0, l-1]
            coord[1, l] = coord[1, l-1] + 1
            coord[2, l] = coord[2, l-1]
        elif direction == 3:  # -y
            coord[0, l] = coord[0, l-1]
            coord[1, l] = coord[1, l-1] - 1
            coord[2, l] = coord[2, l-1]
        elif direction == 4:  # +z
            coord[0, l] = coord[0, l-1]
            coord[1, l] = coord[1, l-1]
            coord[2, l] = coord[2, l-1] + 1
        elif direction == 5:  # -z
            coord[0, l] = coord[0, l-1]
            coord[1, l] = coord[1, l-1]
            coord[2, l] = coord[2, l-1] - 1
    
    return coord

def initial_SAW(N: int):  # Dimerisation
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
            translated = (first[:, -1].reshape(3, 1) + second).T.T
            concatenated = np.concatenate((first, translated[:, 1:]), axis=1)
            tmp = check_self_avoidance(concatenated)

        return concatenated
    
def initial_SAW_rod(N: int):
    coord = np.zeros((3, N), dtype=int)
    for i in range(N):
        coord[0, i] = i
    return coord

# Define the rotation matrices for 3D
def get_3d_rotation_matrix(g):
    # Basic 90-degree rotations around x, y, and z axes
    R_x90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    R_y90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # For simplicity, let's implement a subset of the 24 proper rotations
    # and combine them with reflections for variety
    rotations = [
        np.eye(3),  # identity
        R_x90,  # 90° around x
        np.dot(R_x90, R_x90),  # 180° around x
        np.dot(np.dot(R_x90, R_x90), R_x90),  # 270° around x
        R_y90,  # 90° around y
        np.dot(R_y90, R_y90),  # 180° around y
        np.dot(np.dot(R_y90, R_y90), R_y90),  # 270° around y
        R_z90,  # 90° around z
        np.dot(R_z90, R_z90),  # 180° around z
        np.dot(np.dot(R_z90, R_z90), R_z90),  # 270° around z
        # Add some compound rotations
        np.dot(R_x90, R_y90),  # Combined rotation
        np.dot(R_y90, R_z90),  # Combined rotation
        np.dot(R_z90, R_x90),  # Combined rotation
        np.dot(np.dot(R_x90, R_y90), R_z90),  # Combined rotation
        np.dot(np.dot(R_z90, R_y90), R_x90),  # Combined rotation
        np.dot(np.dot(R_y90, R_x90), R_z90),  # Combined rotation
        # Add reflections
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # reflection in yz plane
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # reflection in xz plane
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # reflection in xy plane
        # Add more if needed, up to the 48 possible symmetry operations
    ]
    
    return rotations[g % len(rotations)]

def symmetry_op(coord):
    N = np.size(coord, axis=1)
    # Choose a random operation from our symmetry group
    g = np.random.randint(0, 19)  # 19 operations defined above
    
    rotation_matrix = get_3d_rotation_matrix(g)
    
    # Apply the rotation matrix to each point
    for i in range(N):
        coord[:, i] = np.dot(rotation_matrix, coord[:, i])
            
    return coord, g

def pivot_step(coord):
    global accepted
    N = np.size(coord, axis=1)
    pivot = np.random.randint(1, N)
    to_be_operated = coord[:, pivot:]
    to_be_operated = (to_be_operated.T - to_be_operated[:, 0]).T
    operated, g = symmetry_op(to_be_operated)
    
    if not hasattr(pivot_step, 'accepted'):
        # Initialize accepted on first call
        pivot_step.accepted = np.zeros((2, 19))  # 19 operations defined
    
    pivot_step.accepted[1, g] += 1
    
    operated = (operated.T + coord[:, pivot]).T
    
    coord_new = np.concatenate((coord[:, :pivot], operated), axis=1)
    not_selfavoidant = check_self_avoidance(coord_new, pivot)
    
    if not_selfavoidant:
        return coord
    else:
        pivot_step.accepted[0, g] += 1
        return coord_new

def squared_end_to_end(coord):
    return np.linalg.norm(coord[:, -1])**2

def squared_gyration(coord):
    N = np.size(coord, axis=1)
    a = (1/N * np.sum(coord, axis=1))
    inbracket = np.dot(a, a)
    result = 0
    for i in range(N):
        result += np.dot(coord[:, i], coord[:, i]) - inbracket
    return 1/N * result

def pivot_run(coord, iterations):
    if not hasattr(pivot_step, 'accepted'):
        pivot_step.accepted = np.zeros((2, 19))  # Initialize if not already done
        
    ω_squared = []
    S_squared = []
    for i in tqdm(range(iterations)):
        coord = pivot_step(coord)
        ω_squared.append(squared_end_to_end(coord))
        S_squared.append(squared_gyration(coord))
    return ω_squared, S_squared, pivot_step.accepted

def autocorrelation(timeseries, mean=None):
    timeseries = np.array(timeseries)
    if mean is None:
        mean = timeseries.mean()
    fluctuations = timeseries - mean

    C = np.fft.ifft(np.fft.fft(fluctuations) * np.fft.ifft(fluctuations)).real
    return C/C[0]

def integrated_autocorrelation_time(timeseries, mean=None, until=None):
    steps = len(timeseries)
    if until is None:
        until = steps // 2
    Gamma = autocorrelation(timeseries, mean=mean)[:until]
    try:
        first_zero = np.where(Gamma <= 0)[0][0]
        print("First zero = ", first_zero)
        return 0.5 + Gamma[1:first_zero].sum()
    except:
        # Gamma never hits 0. So the autocorrelation spans ~ the whole ensemble.
        return steps  # at least!
    
def binning_analyis(data, kmax):
    M = len(data)
    error_est = []
    for k in range(1, kmax+1):
        M_k = M//k
        X_k = np.mean(np.reshape(data[:M_k*k], (-1, k)).T, axis=0)
        error_est.append(np.std(X_k)/np.sqrt(M_k))
    τ = 0.5*((error_est[-1]/error_est[0])**2-1)
    return error_est, int(np.ceil(τ))

def plot_3d_walk(coord, filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the walk
    ax.plot(coord[0, :], coord[1, :], coord[2, :], 'b-', linewidth=2)
    
    # Highlight start and end points
    ax.scatter(coord[0, 0], coord[1, 0], coord[2, 0], color='green', s=100, label='Start')
    ax.scatter(coord[0, -1], coord[1, -1], coord[2, -1], color='red', s=100, label='End')
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.zaxis.set_major_locator(MultipleLocator(1))
    
    plt.title('3D Self-Avoiding Walk')
    plt.legend()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate initial SAW
    N = 10  # Length of the walk
    saw = initial_SAW(N)
    
    # Plot initial configuration
    plot_3d_walk(saw, 'initial_3d_saw.png')
    
    # Run the pivot algorithm
    iterations = 100
    ω_squared, S_squared, accepted = pivot_run(saw, iterations)
    
    # Plot final configuration
    saw_final = pivot_step(saw)
    plot_3d_walk(saw_final, 'final_3d_saw.png')
    
    # Plot observables
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(ω_squared)
    plt.title('End-to-End Distance Squared')
    plt.xlabel('Iteration')
    plt.ylabel('R²')
    
    plt.subplot(122)
    plt.plot(S_squared)
    plt.title('Radius of Gyration Squared')
    plt.xlabel('Iteration')
    plt.ylabel('S²')
    
    plt.tight_layout()
    plt.savefig('observables_3d.png')
    plt.show()
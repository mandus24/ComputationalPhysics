import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import time
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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
    
    # implement a subset of the 24 proper rotations + reflections

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

# NEW FUNCTION: Calculate interaction energy between non-bonded monomers
def calculate_interaction_energy(coords, interaction_strength=1.0, interaction_range=2.0, lennard_jones=True):
    N = coords.shape[1]
    energy = 0.0
    
    # Check all non-bonded pairs
    for i in range(N):
        for j in range(i+2, N):  # Skip bonded neighbors
            r_vec = coords[:, i] - coords[:, j]
            distance = np.linalg.norm(r_vec)
            
            # Apply interaction if within range
            if distance <= interaction_range:
                if lennard_jones:
                    # Lennard-Jones potential: 4ε[(σ/r)^12 - (σ/r)^6]
                    # Using σ=1 for simplicity
                    sigma = 1.0
                    epsilon = abs(interaction_strength)
                    sign = np.sign(interaction_strength)
                    
                    energy += sign * 4 * epsilon * ((sigma/distance)**12 - (sigma/distance)**6)
                else:
                    # Simple power law potential
                    energy += interaction_strength / (distance**6)
    
    return energy

# Calculate bending energy
def calculate_bending_energy(coords, stiffness=1.0):
    N = coords.shape[1]
    energy = 0.0
    
    for i in range(1, N-1):
        v1 = coords[:, i] - coords[:, i-1]
        v2 = coords[:, i+1] - coords[:, i]
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:  
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            cos_angle = np.dot(v1, v2)
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            
            energy += stiffness * (1 - cos_angle)
    
    return energy

# Calculate total energy
def calculate_total_energy(coords, interaction_strength=1.0, interaction_range=2.0, 
                          stiffness=1.0, use_lennard_jones=False):
    interaction_e = calculate_interaction_energy(
        coords, interaction_strength, interaction_range, use_lennard_jones
    )
    bending_e = calculate_bending_energy(coords, stiffness)
    
    return interaction_e + bending_e

# Modified pivot step with Metropolis criterion
def pivot_step_with_metropolis(coord, temperature=1.0, interaction_strength=1.0, 
                              interaction_range=2.0, stiffness=1.0, use_lennard_jones=False):
    N = np.size(coord, axis=1)
    pivot = np.random.randint(1, N-1)  # Avoid selecting the very last monomer as pivot
    
    # Current energy
    energy_before = calculate_total_energy(
        coord, interaction_strength, interaction_range, 
        stiffness, use_lennard_jones
    )
    
    # Perform the pivot
    to_be_operated = coord[:, pivot:]
    to_be_operated = (to_be_operated.T - to_be_operated[:, 0]).T
    operated, g = symmetry_op(to_be_operated)
    operated = (operated.T + coord[:, pivot]).T
    
    coord_new = np.concatenate((coord[:, :pivot], operated), axis=1)
    
    # Check self-avoidance first (hard constraint)
    if check_self_avoidance(coord_new, pivot):
        return coord, False, energy_before
    
    # Calculate new energy
    energy_after = calculate_total_energy(
        coord_new, interaction_strength, interaction_range, 
        stiffness, use_lennard_jones
    )
    
    # Metropolis acceptance criterion
    delta_energy = energy_after - energy_before
    accepted = False
    
    if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / temperature):
        accepted = True
        return coord_new, accepted, energy_after
    else:
        return coord, accepted, energy_before

# Run simulation with energy calculations
def pivot_run_with_energy(coord, iterations, temperature=1.0, interaction_strength=1.0, 
                         interaction_range=2.0, stiffness=1.0, use_lennard_jones=False):
    if not hasattr(pivot_run_with_energy, 'acceptance_stats'):
        pivot_run_with_energy.acceptance_stats = {'attempts': 0, 'accepted': 0}
    
    ω_squared = []  # End-to-end distance squared
    S_squared = []  # Radius of gyration squared
    energies = []   # Total energy
    acceptance_rate = []  # Acceptance rate over time
    
    for i in tqdm(range(iterations)):
        coord, accepted, energy = pivot_step_with_metropolis(
            coord, temperature, interaction_strength, 
            interaction_range, stiffness, use_lennard_jones
        )
        
        # Update statistics
        pivot_run_with_energy.acceptance_stats['attempts'] += 1
        if accepted:
            pivot_run_with_energy.acceptance_stats['accepted'] += 1
        
        # Calculate observables
        ω_squared.append(squared_end_to_end(coord))
        S_squared.append(squared_gyration(coord))
        energies.append(energy)
        
        # Calculate running acceptance rate
        acceptance_rate.append(
            pivot_run_with_energy.acceptance_stats['accepted'] / 
            pivot_run_with_energy.acceptance_stats['attempts']
        )
    
    return ω_squared, S_squared, energies, acceptance_rate, coord

def squared_end_to_end(coord):
    return np.linalg.norm(coord[:, -1] - coord[:, 0])**2

def squared_gyration(coord):
    N = np.size(coord, axis=1)
    a = (1/N * np.sum(coord, axis=1))
    inbracket = np.dot(a, a)
    result = 0
    for i in range(N):
        result += np.dot(coord[:, i], coord[:, i]) - inbracket
    return 1/N * result

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

# NEW FUNCTION: Enhanced 3D walk plotting with interaction energy coloring
def plot_3d_walk_with_energy(coord, filename=None, interaction_strength=1.0, 
                            interaction_range=2.0, colorful=True):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    N = coord.shape[1]
    
    if colorful:
        # Calculate local interaction energy for each monomer
        local_energy = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if abs(i-j) > 1:  # Non-bonded
                    r_vec = coord[:, i] - coord[:, j]
                    distance = np.linalg.norm(r_vec)
                    if distance <= interaction_range:
                        local_energy[i] += interaction_strength / (distance**6)
        
        # Normalize energies for coloring
        if np.max(local_energy) != np.min(local_energy):
            norm = Normalize(vmin=np.min(local_energy), vmax=np.max(local_energy))
            colors = cm.viridis(norm(local_energy))
        else:
            colors = cm.viridis(np.zeros(N))
        
        # Plot segments with colors
        for i in range(N-1):
            ax.plot(coord[0, i:i+2], coord[1, i:i+2], coord[2, i:i+2], 
                   color=colors[i], linewidth=3)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cm.viridis)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('Local Interaction Energy')
    else:
        # Standard plot
        ax.plot(coord[0, :], coord[1, :], coord[2, :], 'b-', linewidth=2)
    
    # Highlight start and end points
    ax.scatter(coord[0, 0], coord[1, 0], coord[2, 0], color='green', s=100, label='Start')
    ax.scatter(coord[0, -1], coord[1, -1], coord[2, -1], color='red', s=100, label='End')
    
    # Add spheres at each monomer position
    for i in range(N):
        ax.scatter(coord[0, i], coord[1, i], coord[2, i], color='black', alpha=0.5, s=50)
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    max_range = np.array([
        coord[0, :].max() - coord[0, :].min(),
        coord[1, :].max() - coord[1, :].min(),
        coord[2, :].max() - coord[2, :].min()
    ]).max() / 2.0
    
    mid_x = (coord[0, :].max() + coord[0, :].min()) * 0.5
    mid_y = (coord[1, :].max() + coord[1, :].min()) * 0.5
    mid_z = (coord[2, :].max() + coord[2, :].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    title = "3D Polymer Simulation"
    if interaction_strength > 0:
        title += " (Attractive)"
    elif interaction_strength < 0:
        title += " (Attractive)"
    plt.title(title)
    plt.legend()
    
    plt.show()

# NEW FUNCTION: Compare different polymer configurations
def compare_polymer_configurations(configs, labels, filename=None, view_angle=(30, 30)):
    n_configs = len(configs)
    fig = plt.figure(figsize=(5*n_configs, 10))
    
    for i in range(n_configs):
        ax = fig.add_subplot(1, n_configs, i+1, projection='3d')
        
        coord = configs[i]
        N = coord.shape[1]
        
        # Plot the walk
        ax.plot(coord[0, :], coord[1, :], coord[2, :], 'b-', linewidth=2)
        
        # Highlight start and end points
        ax.scatter(coord[0, 0], coord[1, 0], coord[2, 0], color='green', s=100, label='Start')
        ax.scatter(coord[0, -1], coord[1, -1], coord[2, -1], color='red', s=100, label='End')
        
        # Add spheres at each monomer position
        for j in range(N):
            ax.scatter(coord[0, j], coord[1, j], coord[2, j], color='black', alpha=0.5, s=30)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([
            coord[0, :].max() - coord[0, :].min(),
            coord[1, :].max() - coord[1, :].min(),
            coord[2, :].max() - coord[2, :].min()
        ]).max() / 2.0
        
        mid_x = (coord[0, :].max() + coord[0, :].min()) * 0.5
        mid_y = (coord[1, :].max() + coord[1, :].min()) * 0.5
        mid_z = (coord[2, :].max() + coord[2, :].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set view angle to be the same for all subplots
        ax.view_init(view_angle[0], view_angle[1])
        
        ax.set_title(labels[i])
    
    plt.tight_layout()
    
    plt.show()

# NEW FUNCTION: Plot simulation observables
def plot_simulation_results(ω_squared, S_squared, energies, acceptance_rate, filename=None):
    fig = plt.figure(figsize=(12, 10))
    
    # End-to-end distance squared
    ax1 = fig.add_subplot(221)
    ax1.plot(ω_squared)
    ax1.set_title('End-to-End Distance Squared')
    ax1.set_xlabel('MC Steps')
    ax1.set_ylabel('R²')
    ax1.grid(True, alpha=0.3)
    
    # Radius of gyration squared
    ax2 = fig.add_subplot(222)
    ax2.plot(S_squared)
    ax2.set_title('Radius of Gyration Squared')
    ax2.set_xlabel('MC Steps')
    ax2.set_ylabel('S²')
    ax2.grid(True, alpha=0.3)
    
    # Energy
    ax3 = fig.add_subplot(223)
    ax3.plot(energies)
    ax3.set_title('Total Energy')
    ax3.set_xlabel('MC Steps')
    ax3.set_ylabel('Energy')
    ax3.grid(True, alpha=0.3)
    
    # Acceptance rate
    ax4 = fig.add_subplot(224)
    ax4.plot(acceptance_rate)
    ax4.set_title('Acceptance Rate')
    ax4.set_xlabel('MC Steps')
    ax4.set_ylabel('Rate')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()

    
    plt.show()

# NEW FUNCTION: Plot energy distributions
def plot_energy_distribution(neutral_energies, Repulsive_energies, Attractive_energies, filename=None):
    plt.figure(figsize=(10, 6))
    
    plt.hist(neutral_energies, bins=30, alpha=0.5, label='Neutral', color='blue')
    plt.hist(Repulsive_energies, bins=30, alpha=0.5, label='Repulsive', color='green')
    plt.hist(Attractive_energies, bins=30, alpha=0.5, label='Attractive', color='red')
    
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.title('Energy Distribution for Different Interaction Types')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Set random seed for reproducibility
np.random.seed(42)

# Set parameters
polymer_length = 30
iterations = 1000
temperature = 1.0

# Create initial polymer configurations
print("Generating initial self-avoiding walk...")
polymer = initial_SAW(polymer_length)

# Create various polymers with different interaction parameters
print("Running simulations with different interaction energies...")

# Neutral (stiffness only)
print("Running neutral simulation...")
omega_sq_neutral, gyration_sq_neutral, energies_neutral, acceptance_neutral, final_polymer_neutral = pivot_run_with_energy(
    polymer.copy(), 
    iterations=iterations,
    temperature=temperature,
    interaction_strength=0.0,  # Neutral (no non-bonded interactions)
    interaction_range=2.0,
    stiffness=1.0
)

# Repulsive interactions
print("Running Attractive simulation...")
omega_sq_Repulsive, gyration_sq_Repulsive, energies_Repulsive, acceptance_Repulsive, final_polymer_Repulsive = pivot_run_with_energy(
    polymer.copy(),
    iterations=iterations,
    temperature=temperature,
    interaction_strength=-1.0,  # Negative for attraction
    interaction_range=2.0,
    stiffness=1.0
)

# Attractive interactions
print("Running Repulsive simulation...")
omega_sq_Attractive, gyration_sq_Attractive, energies_Attractive, acceptance_Attractive, final_polymer_Attractive = pivot_run_with_energy(
    polymer.copy(),
    iterations=iterations,
    temperature=temperature,
    interaction_strength=1.0,  # Positive for repulsion
    interaction_range=2.0,
    stiffness=1.0
)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Neutral simulation - Average end-to-end distance: {np.mean(omega_sq_neutral):.2f}")
print(f"Neutral simulation - Average radius of gyration: {np.mean(gyration_sq_neutral):.2f}")
print(f"Neutral simulation - Average energy: {np.mean(energies_neutral):.2f}")
print(f"Neutral simulation - Final acceptance rate: {acceptance_neutral[-1]:.4f}")

print(f"Repulsive simulation - Average end-to-end distance: {np.mean(omega_sq_Repulsive):.2f}")
print(f"Repulsive simulation - Average radius of gyration: {np.mean(gyration_sq_Repulsive):.2f}")
print(f"Repulsive simulation - Average energy: {np.mean(energies_Repulsive):.2f}")
print(f"Repulsive simulation - Final acceptance rate: {acceptance_Repulsive[-1]:.2f}")

print(f"Attractive simulation - Average end-to-end distance: {np.mean(omega_sq_Attractive):.2f}")
print(f"Attractive simulation - Average radius of gyration: {np.mean(gyration_sq_Attractive):.2f}")
print(f"Attractive simulation - Average energy: {np.mean(energies_Attractive):.2f}")
print(f"Attractive simulation - Final acceptance rate: {acceptance_Attractive[-1]:.2f}")

# Plot 3D visualizations of the final configurations
print("\nGenerating 3D visualizations...")
plot_3d_walk_with_energy(
    final_polymer_neutral, 
    interaction_strength=2.0,
    interaction_range=2.0
)

plot_3d_walk_with_energy(
    final_polymer_Repulsive, 
    interaction_strength=-1.0,
    interaction_range=2.0
)

plot_3d_walk_with_energy(
    final_polymer_Attractive, 
    interaction_strength=1.0,
    interaction_range=2.0
)#

# Compare all three configurations side by side
compare_polymer_configurations(
    [final_polymer_neutral, final_polymer_Repulsive, final_polymer_Attractive],
    ["Neutral", "Repulsive", "Attractive"]
)

# Plot simulation statistics
plot_simulation_results(
    omega_sq_neutral, 
    gyration_sq_neutral, 
    energies_neutral, 
    acceptance_neutral
)

# Plot simulation statistics
plot_simulation_results(
    omega_sq_Repulsive, 
    gyration_sq_Repulsive, 
    energies_Repulsive, 
    acceptance_Repulsive
)

# Plot simulation statistics
plot_simulation_results(
    omega_sq_Attractive, 
    gyration_sq_Attractive, 
    energies_Attractive, 
    acceptance_Attractive
)

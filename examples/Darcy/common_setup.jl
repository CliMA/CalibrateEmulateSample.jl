### Setup params
# Define the spatial domain and discretization 
dim = 2
N, L = 160, 1.0
pts_per_dim = LinRange(0, L, N)
obs_ΔN = 5

# Define GRF parameters
smoothness = 0.1
corr_length = 0.7
dofs = 40

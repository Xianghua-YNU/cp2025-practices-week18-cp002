import numpy as np
import matplotlib.pyplot as plt

# Define the integrand function
def f(x):
    return x**(-1/2) / (np.exp(x) + 1)

# Define the weight function p(x) = 1/(2*sqrt(x))
def p(x):
    return 1 / (2 * np.sqrt(x))

# 1. Generate random numbers following the weight function p(x) distribution
def generate_random_numbers(size):
    # Using inverse transform method to generate random numbers following p(x)
    # The CDF of p(x) is F(x) = sqrt(x)
    # Inverse transform: x = F^{-1}(u) = u^2
    u = np.random.uniform(0, 1, size)
    return u**2

# 2. Calculate the integral using importance sampling
def importance_sampling_integration(N):
    # Generate random numbers following p(x) distribution
    x_samples = generate_random_numbers(N)
    
    # Calculate f(x)/p(x)
    f_over_p = f(x_samples) / p(x_samples)
    
    # Estimate the integral value
    integral_estimate = np.mean(f_over_p)
    
    # Calculate variance and statistical error
    var_f = np.var(f_over_p)
    statistical_error = np.sqrt(var_f) / np.sqrt(N)
    
    return integral_estimate, statistical_error

# Set random seed for reproducibility
np.random.seed(42)

# Calculate the integral
N = 1000000
integral_estimate, error = importance_sampling_integration(N)

# Output results
print(f"Integral estimate: {integral_estimate:.6f}")
print(f"Statistical error: {error:.6f}")
print(f"95% confidence interval: [{integral_estimate - 1.96*error:.6f}, {integral_estimate + 1.96*error:.6f}]")

# Optional: Visualize convergence
sample_sizes = np.logspace(3, 6, 50).astype(int)
estimates = []
errors = []

for size in sample_sizes:
    est, err = importance_sampling_integration(size)
    estimates.append(est)
    errors.append(err)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, estimates, 'b-', label='Integral estimate')
plt.axhline(y=0.84, color='r', linestyle='--', label='Reference value 0.84')
plt.fill_between(sample_sizes, 
                 np.array(estimates) - np.array(errors), 
                 np.array(estimates) + np.array(errors),
                 color='b', alpha=0.2, label='Error range')
plt.xscale('log')
plt.xlabel('Sample size (N)')
plt.ylabel('Integral estimate')
plt.title('Importance Sampling Convergence')
plt.legend()
plt.grid(True)
plt.show()

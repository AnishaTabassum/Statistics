import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from scipy.stats import norm

# -------------------------------------------------
# STEP 1: Generate sample data
# -------------------------------------------------
# Generate 10,000 random samples from a Normal distribution
# loc = mean (μ = 50)
# scale = standard deviation (σ = 5)
sample = normal(loc=50, scale=5, size=10000)

# -------------------------------------------------
# STEP 2: Compute sample statistics
# -------------------------------------------------
# Estimate the parameters (mean and standard deviation)
# from the generated data (Maximum Likelihood Estimation)
sample_mean = sample.mean()
sample_std = sample.std()

# -------------------------------------------------
# STEP 3: Fit a Normal distribution using estimated parameters
# -------------------------------------------------
# Create a normal distribution object using estimated mean and std
# This is the parametric model assumption:
# "Data follows a Gaussian distribution"
dist = norm(sample_mean, sample_std)

# -------------------------------------------------
# STEP 4: Generate x-values for smooth PDF curve
# -------------------------------------------------
# Create evenly spaced values between minimum and maximum sample value
values = np.linspace(sample.min(), sample.max(), 100)

# Compute PDF (Probability Density Function) values
# for each x value
density_probabilities = [dist.pdf(value) for value in values]

# -------------------------------------------------
# STEP 5: Plot Histogram and Fitted PDF
# -------------------------------------------------
# Plot normalized histogram (density=True ensures area = 1)
plt.hist(sample, bins=10, density=True)

# Plot the estimated Gaussian PDF curve
plt.plot(values, density_probabilities)

plt.title("Parametric Density Estimation (Normal Distribution)")
plt.xlabel("Value")
plt.ylabel("Density")

plt.show()

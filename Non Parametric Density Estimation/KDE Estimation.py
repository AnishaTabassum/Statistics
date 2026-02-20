import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from sklearn.neighbors import KernelDensity

# -------------------------------------------------
# STEP 1: Generate synthetic data (Bimodal Distribution)
# -------------------------------------------------
# Generate 300 samples from Normal(20, 5)
sample1 = normal(loc=20, scale=5, size=300)

# Generate 700 samples from Normal(40, 5)
sample2 = normal(loc=40, scale=5, size=700)

# Combine both datasets to create a bimodal dataset
sample = np.hstack((sample1, sample2))

# -------------------------------------------------
# STEP 2: Visualize raw data using histogram
# -------------------------------------------------
# This shows the empirical distribution of the data
plt.hist(sample, bins=50)

# -------------------------------------------------
# STEP 3: Apply Kernel Density Estimation (KDE)
# -------------------------------------------------
# KDE is a non-parametric method:
# It does NOT assume a specific distribution (like Normal).
# Instead, it estimates the density using kernels.

# Create KDE model with Gaussian kernel
# bandwidth controls smoothness of the curve
model = KernelDensity(bandwidth=3, kernel='gaussian')

# sklearn requires input data in 2D shape (n_samples, n_features)
sample = sample.reshape(len(sample), 1)

# Fit the KDE model to the data
model.fit(sample)

# -------------------------------------------------
# STEP 4: Generate points for smooth density curve
# -------------------------------------------------
# Create evenly spaced values between min and max
values = np.linspace(sample.min(), sample.max(), 100)

# Convert to 2D array for prediction
values = values.reshape(len(values), 1)

# score_samples() returns log-density values
density_probabilities = model.score_samples(values)

# Convert log-density back to normal density
density_probabilities = np.exp(density_probabilities)

# -------------------------------------------------
# STEP 5: Plot normalized histogram and KDE curve
# -------------------------------------------------
plt.hist(sample, bins=50, density=True)  # normalized histogram
plt.plot(values, density_probabilities)  # KDE smooth curve

plt.title("Non-Parametric Density Estimation using KDE")
plt.xlabel("Value")
plt.ylabel("Density")

plt.show()

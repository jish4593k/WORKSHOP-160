import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
from scipy.stats import norm

# Constants
MIN = 1
MAX = 10

# Generate random integer data between MIN and MAX
data = np.random.randint(low=MIN * 100, high=(MAX + 1) * 100, size=(100, 5))
dataf = data.astype(np.float64) / 100

# Standardize the data using scikit-learn StandardScaler
scaler = StandardScaler()
dataf_standardized = scaler.fit_transform(dataf)

# Calculate mean and standard deviation along each column of standardized data
datamean = np.mean(dataf_standardized, axis=0)
datastd = np.std(dataf_standardized, axis=0)

# Create a normal distribution for each column using mean and standard deviation
pds = [norm(loc=mean, scale=std) for mean, std in zip(datamean, datastd)]

# Calculate probabilities for each data point to belong to the given probability distribution
probabilities = np.array([pd.pdf(dataf_standardized[:, i]) for i, pd in enumerate(pds)]).T

# Calculate epsilon values by taking the product along columns
epsilons = np.prod(probabilities, axis=1)

# Plot epsilon values
plt.plot(epsilons)
plt.ylabel('epsilon')
plt.xlabel('sample index')
plt.show()

"""
Back-propagation Neural Net.
"""

# System Imports.
import numpy, pandas

# User Class Imports.
from resources import logging
import neural_net


# Initialize logging.
logger = logging.get_logger(__name__)


# Load in CSV.
housing_data = pandas.read_csv('./Documents/other_housing.csv')

# Initially only work with first 10, for testing purposes.
housing_data = housing_data[0:10]

# Normalize data.
normalizer = neural_net.Normalizer()
normalized_data = normalizer.normalize_data(housing_data)

# Start neural net.
backprop = neural_net.BackPropNet(normalized_data)

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
housing_data = housing_data[0:5]

# Normalize data.
logger.info('')
normalizer = neural_net.Normalizer()
normalized_data = normalizer.normalize_data(housing_data)
features = normalized_data.loc[:, normalized_data.columns != 'SalePrice']
targets = normalized_data['SalePrice']

logger.info('')
logger.info('Normalized Features: \n{0}'.format(features))
logger.info('Normailzed Targets: \n{0}'.format(targets))

# Start neural net.
backprop = neural_net.BackPropNet(normalized_data)

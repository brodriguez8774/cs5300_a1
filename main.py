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

# Initially only work with first few, for testing purposes.
# housing_data = housing_data[0:3]

# Normalize data.
logger.info('')
normalizer = neural_net.Normalizer()
normalized_data = normalizer.normalize_data(housing_data)

# Randomize data order.
randomized_data = normalized_data.iloc[numpy.random.permutation(len(normalized_data))]

features = randomized_data.loc[:, randomized_data.columns != 'SalePrice']
targets = randomized_data['SalePrice']

logger.info('')
# logger.info('Normalized Features: \n{0}'.format(features))
# logger.info('Normalized Targets: \n{0}'.format(targets))

# Start neural net.
backprop = neural_net.BackPropNet(features)

max_index = 0
while max_index < len(features):
    min_index = max_index
    max_index += 20
    training_features = features[min_index:max_index]
    training_targets = targets[min_index:max_index]
    backprop.train(training_features.values, training_targets.values)

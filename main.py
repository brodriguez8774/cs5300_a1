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

# Normalize data.
logger.info('')
normalizer = neural_net.Normalizer()
normalized_data = normalizer.normalize_data(housing_data)

# Randomize data order and split into features/targets.
randomized_data =  normalized_data.iloc[numpy.random.permutation(len(normalized_data))]
features = randomized_data.loc[:, randomized_data.columns != 'SalePrice']
targets = randomized_data['SalePrice']

logger.info('')
# logger.info('Normalized Features: \n{0}'.format(features))
# logger.info('Normalized Targets: \n{0}'.format(targets))


# Artifically limit data size for time purposes during testing.
data_size = 200
randomized_data = randomized_data[0:data_size]
features = features[0:data_size]
targets = targets[0:data_size]


# Start neural net.
backprop = neural_net.BackPropNet(randomized_data)


epochs = 20
data_step = 500
# Repeatedly iterate through full dataset, epoch times.
for index in range(epochs):
    max_index = 0
    error_total = 0
    # Split dataset up into sections and train. Section size determined by data_step.
    while max_index < len(features):
        # Adjust index of training data.
        min_index = max_index
        max_index += data_step
        training_features = features[min_index:max_index]
        training_targets = targets[min_index:max_index]

        # Train using training data.
        error_total = backprop.train(training_features.values, training_targets.values)

    # Print out total error for current epoch.
    logger.info('Epoch {0} Error: {1}'.format(index, error_total))

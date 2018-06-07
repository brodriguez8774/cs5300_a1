"""
Back-propagation Neural Net.
"""

# System Imports.
from matplotlib import patches, pyplot
import numpy, pandas, scipy

# User Class Imports.
from resources import logging
import neural_net, result_tracker

# Initialize logging.
logger = logging.get_logger(__name__)


# Load in CSV.
housing_data = pandas.read_csv('./Documents/other_housing.csv')

# Normalize data.
logger.info('')
normalizer = neural_net.Normalizer()
normalized_data = normalizer.normalize_data(housing_data)
logger.info('Dataset has {0} total rows.'.format(len(normalized_data)))
logger.info('')


def randomize_data(data_size=None):
    """
    Randomize data.
    :param data_size: Limiting size to set data at.
    :return: Randomized full dataset, randomized features, and randomized targets.
    """
    # Artifically limit data size for time purposes during testing.
    if data_size is not None:
        cropped_data = normalized_data[0:data_size]
    else:
        cropped_data = normalized_data

    # Randomize data order and split into features/targets.
    randomized_data = cropped_data.iloc[numpy.random.permutation(len(cropped_data))]
    features = randomized_data.loc[:, randomized_data.columns != 'SalePrice']
    targets = randomized_data['SalePrice']

    # logger.info('Normalized Features: \n{0}'.format(features))
    # logger.info('Normalized Targets: \n{0}'.format(targets))

    return randomized_data, features, targets

# Initial data randomization.
normalized_data, _, _ = randomize_data()



# Initialize variables
epoch_count = 0
epochs = 200         # Max number of times to step through instances of dataset.
data_step = 250     # Number of indexes to step forward while iterating through a single dataset instance.
data_size = 2300     # Total number of records to include within a single dataset instance.

# Initialize result sets.
training_results = []
prediction_results = []

# Start neural nets.
backprop = neural_net.BackPropNet(normalized_data)
backprop_tracker = result_tracker.ResultTracker(epochs/5)

# Repeatedly iterate through full dataset. Continues until either epoch count is met or no further progress is made.
while backprop_tracker.continue_training_check(max_epochs=epochs):
    max_index = 0
    total_error = 0

    # Make sure each epoch has a differently-randomized dataset instance.
    randomized_data, features, targets = randomize_data(data_size=data_size)

    # Split dataset up into sections and train. Section size determined by data_step.
    while max_index < len(features):
        # Adjust index of training data.
        min_index = max_index
        max_index += data_step
        if max_index > len(features):
            max_index = len(features)

        training_features = features[min_index:max_index]
        training_targets = targets[min_index:max_index]

        # Train using training data.
        weights, error = backprop.train(training_features.values, training_targets.values, learn_rate=0.001)
        total_error += error

    # Print out total error for current epoch.
    backprop_tracker.add_iteration(weights, total_error)
    logger.info('Epoch {0} Total Error: {1}'.format(epoch_count, total_error))
    training_results.append([total_error, epoch_count])
    epoch_count += 1


# Plot accuracy of training, over time.
training_numpy_array = numpy.asarray(training_results)
prediction_numpy_array = numpy.asarray(prediction_results)
x = []
y = []

for result in training_results:
    pyplot.scatter(result[1], result[0], alpha=1, c='b')
    x.append(result[1])
    y.append(result[0])

# Plot labels.
pyplot.title('Network Error Results')
pyplot.xlabel('Epoch Number')
pyplot.ylabel('Epoch Error')

# Create best-fit line.
# Borrowed from https://stackoverflow.com/questions/22239691/code-for-line-of-best-fit-of-a-scatter-plot-in-python.
pyplot.plot(numpy.unique(x), numpy.poly1d(numpy.polyfit(x, y, 1))(numpy.unique(x)), 'g', alpha=0.75)
label_1 = patches.Patch(color='g', label='Best Fit Line')
pyplot.legend(handles=[label_1])

pyplot.show()


# # Dillon's implementation of backprop.
# tensor_backprop = neural_net.TensorBackProp(normalized_data)
# tensor_backprop.train()
# tensor_backprop = None


logger.info('Exiting program.')
